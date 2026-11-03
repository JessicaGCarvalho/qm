import { canonicalPerson, foldPrincipalId, personKey } from "../directory/person.ts";
import { createTtlRefresh } from "../util/async.ts";
import { reportFailure } from "../util/errors.ts";
import type { PrincipalRecord, PrincipalSource, PrincipalStore } from "./principal-store.ts";

export type { PrincipalRecord } from "./principal-store.ts";

export interface SeenPerson {
  principalId: string;
  source: PrincipalSource;
  at: number;
}

export interface PrincipalRoster {
  list(): Promise<PrincipalRecord[]>;
  get(principalId: string): Promise<PrincipalRecord | null>;
  note(principalId: string, source: PrincipalSource): Promise<void>;
  record(people: readonly SeenPerson[]): Promise<number>;
  markOffboarded(principalId: string, by: string): Promise<PrincipalRecord>;
  markReactivated(principalId: string): Promise<PrincipalRecord | null>;
}

const earliest = (a: PrincipalRecord, b: PrincipalRecord, at: (r: PrincipalRecord) => number | undefined) =>
  [a, b].filter((r) => at(r) !== undefined).sort((x, y) => at(x)! - at(y)!)[0];

function merge(a: PrincipalRecord, b: PrincipalRecord): PrincipalRecord {
  const primary = foldPrincipalId(b.principalId) === personKey(b.principalId) ? b : a;
  const first = earliest(a, b, (r) => r.firstSeenAt);
  const off = earliest(a, b, (r) => r.offboardedAt);
  return {
    principalId: primary.principalId,
    ...(first ? { firstSeenAt: first.firstSeenAt, source: first.source } : {}),
    ...(off ? { offboardedAt: off.offboardedAt, offboardedBy: off.offboardedBy } : {}),
  };
}

export function createPrincipalRoster(
  store: PrincipalStore,
  opts: { principalLinks?: { refresh(force?: boolean): Promise<void> } } = {},
): PrincipalRoster {
  const rows = new Map<string, PrincipalRecord>();
  const people = new Map<string, PrincipalRecord>();
  const rebuild = () => {
    people.clear();
    for (const r of rows.values()) {
      const key = personKey(r.principalId);
      const other = people.get(key);
      people.set(key, other ? merge(other, r) : r);
    }
  };
  const remember = (record: PrincipalRecord): PrincipalRecord => {
    rows.set(record.principalId, record);
    rebuild();
    return people.get(personKey(record.principalId)) ?? record;
  };
  const refresh = createTtlRefresh(10_000, async () => {
    await opts.principalLinks?.refresh();
    const records = await store.all();
    rows.clear();
    for (const r of records) rows.set(r.principalId, r);
    rebuild();
  });
  const rowId = (principalId: string): string => foldPrincipalId(canonicalPerson(principalId));
  const seen = (principalId: string) => people.get(personKey(principalId))?.firstSeenAt !== undefined;

  async function record(input: readonly SeenPerson[]): Promise<number> {
    const first = new Map<string, SeenPerson>();
    for (const person of input) {
      const id = rowId(person.principalId);
      if (!id || seen(person.principalId)) continue;
      const known = first.get(id);
      if (!known || person.at < known.at) first.set(id, { ...person, principalId: id });
    }
    const added = await store.fillFirstSeen([...first.values()]);
    for (const r of added) remember(r);
    return added.length;
  }

  return {
    async list() {
      await refresh();
      return [...people.values()];
    },
    async get(principalId) {
      await refresh();
      return people.get(personKey(principalId)) ?? null;
    },
    async note(principalId, source) {
      if (seen(principalId)) return;
      await record([{ principalId, source, at: Date.now() }]).catch((err: unknown) =>
        reportFailure("roster: record person", err),
      );
    },
    record,
    async markOffboarded(principalId, by) {
      await refresh(true);
      return remember(await store.markOffboarded(rowId(principalId), by, Date.now()));
    },
    async markReactivated(principalId) {
      await refresh(true);
      const ids = [...rows.values()].filter((r) => personKey(r.principalId) === personKey(principalId));
      let last: PrincipalRecord | null = null;
      for (const r of ids) {
        const cleared = await store.clearOffboarded(r.principalId);
        if (cleared) last = remember(cleared);
      }
      return last;
    },
  };
}
