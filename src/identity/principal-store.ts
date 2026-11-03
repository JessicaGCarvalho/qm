import { createPgPool, type PgMigrationDefinition } from "../persistence/pg-pool.ts";
import { DIRECTORY_MIGRATIONS } from "../directory/postgres-directory-store.ts";
import { SESSION_MIGRATIONS } from "../sessions/session-schema.ts";
import { msOf } from "../sessions/items.ts";

export type PrincipalSource = "turn" | "invite" | "admin";

export interface PrincipalRecord {
  principalId: string;
  firstSeenAt?: number;
  source?: PrincipalSource;
  offboardedAt?: number;
  offboardedBy?: string;
}

export interface PrincipalStore {
  all(): Promise<PrincipalRecord[]>;
  fillFirstSeen(
    people: readonly { principalId: string; source: PrincipalSource; at: number }[],
  ): Promise<PrincipalRecord[]>;
  markOffboarded(principalId: string, by: string, at: number): Promise<PrincipalRecord>;
  clearOffboarded(principalId: string): Promise<PrincipalRecord | null>;
}

export const PRINCIPAL_MIGRATIONS: PgMigrationDefinition[] = [
  {
    id: "users/principals/0001",
    statements: [
      `CREATE TABLE IF NOT EXISTS principals(
        principal_id  TEXT PRIMARY KEY CHECK (principal_id <> ''),
        first_seen_at TIMESTAMPTZ,
        source        TEXT CHECK (source IN ('turn', 'invite', 'admin')),
        offboarded_at TIMESTAMPTZ,
        offboarded_by TEXT
      )`,
    ],
  },
  {
    id: "users/principals/0002-turn-backfill",
    statements: [
      `INSERT INTO principals(principal_id, first_seen_at, source)
       SELECT CASE WHEN position('@' IN btrim(e.author)) > 0 THEN lower(btrim(e.author)) ELSE btrim(e.author) END,
              MIN(e.created_at), 'turn'
         FROM items e
         JOIN sessions s ON s.id = e.session_id
        WHERE e.tenant_id = qm_tenant_id()
          AND e.author IS NOT NULL AND e.author NOT LIKE 'system:%'
          AND e.kind IN ('UserMessage', 'SteeringMessage')
          AND NOT (e.body @> '{"overheard":true}')
          AND s.origin = 'conversation' AND s.parent_session_id IS NULL
        GROUP BY 1
       ON CONFLICT (principal_id) DO NOTHING`,
    ],
  },
];

const COLUMNS = `principal_id, ${msOf("first_seen_at")} AS first_seen_at, source, ${msOf("offboarded_at")} AS offboarded_at, offboarded_by`;
const ts = (param: string): string => `to_timestamp(${param}::double precision / 1000)`;

function toRecord(r: Record<string, unknown>): PrincipalRecord {
  return {
    principalId: r.principal_id as string,
    ...(r.first_seen_at != null ? { firstSeenAt: Number(r.first_seen_at), source: r.source as PrincipalSource } : {}),
    ...(r.offboarded_at != null
      ? { offboardedAt: Number(r.offboarded_at), offboardedBy: r.offboarded_by as string }
      : {}),
  };
}

export function createPostgresPrincipalStore(connectionString: string): PrincipalStore {
  const { q } = createPgPool(connectionString, [
    ...DIRECTORY_MIGRATIONS,
    ...SESSION_MIGRATIONS,
    ...PRINCIPAL_MIGRATIONS,
  ]);
  return {
    async all() {
      return (await q(`SELECT ${COLUMNS} FROM principals`)).map(toRecord);
    },
    async fillFirstSeen(people) {
      if (!people.length) return [];
      const rows = await q(
        `INSERT INTO principals AS p(principal_id, first_seen_at, source)
         SELECT id, ${ts("at")}, source FROM unnest($1::text[], $2::bigint[], $3::text[]) AS t(id, at, source)
         ON CONFLICT (principal_id) DO UPDATE SET first_seen_at = EXCLUDED.first_seen_at, source = EXCLUDED.source
           WHERE p.first_seen_at IS NULL
         RETURNING ${COLUMNS}`,
        [people.map((p) => p.principalId), people.map((p) => p.at), people.map((p) => p.source)],
      );
      return rows.map(toRecord);
    },
    async markOffboarded(principalId, by, at) {
      const rows = await q(
        `INSERT INTO principals AS p(principal_id, offboarded_at, offboarded_by) VALUES ($1, ${ts("$3")}, $2)
         ON CONFLICT (principal_id) DO UPDATE SET offboarded_at = COALESCE(p.offboarded_at, EXCLUDED.offboarded_at),
           offboarded_by = CASE WHEN p.offboarded_at IS NULL THEN EXCLUDED.offboarded_by ELSE p.offboarded_by END
         RETURNING ${COLUMNS}`,
        [principalId, by, at],
      );
      return toRecord(rows[0]!);
    },
    async clearOffboarded(principalId) {
      const rows = await q(
        `UPDATE principals SET offboarded_at = NULL, offboarded_by = NULL WHERE principal_id = $1 RETURNING ${COLUMNS}`,
        [principalId],
      );
      return rows[0] ? toRecord(rows[0]) : null;
    },
  };
}
