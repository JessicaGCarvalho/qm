import type { AdminGrant } from "./admin-grant-store.ts";
import { adminStatusFromGrants, type AdminStatus } from "./admin-service.ts";
import { forEachAttributedTurn, type AttributionInput } from "./attribution.ts";
import { canonicalPerson } from "../directory/person.ts";
import { isTeammate, type ExternalMember } from "../identity/external-members.ts";
import type { PrincipalRoster, SeenPerson } from "../identity/principal-roster.ts";

export interface AdminUserRow {
  principalId: string;
  sessionCount: number;
  turnCount: number;
  lastSeenAt: number | null;
  admin: AdminStatus;
}

export interface UsersInput extends AttributionInput {
  grants: readonly AdminGrant[];
  people: readonly string[];
}

export function computeUsers(input: UsersInput): AdminUserRow[] {
  const { grants, people } = input;

  const sessionsByUser = new Map<string, Set<string>>();
  const turnsByUser = new Map<string, number>();
  const lastSeenByUser = new Map<string, number>();
  const bumpLastSeen = (principalId: string, t: number) => {
    if (t > (lastSeenByUser.get(principalId) ?? 0)) lastSeenByUser.set(principalId, t);
  };

  forEachAttributedTurn(input, {
    onWindow(sessionId, w) {
      const principalId = canonicalPerson(w.principalId);
      const set = sessionsByUser.get(principalId);
      if (set) set.add(sessionId);
      else sessionsByUser.set(principalId, new Set([sessionId]));
      bumpLastSeen(principalId, w.validFrom);
    },
    onTurn(w, turn) {
      const principalId = canonicalPerson(w.principalId);
      turnsByUser.set(principalId, (turnsByUser.get(principalId) ?? 0) + turn.turns);
      bumpLastSeen(principalId, turn.lastAt);
    },
  });

  const ids = new Set(people.map(canonicalPerson));
  return [...ids]
    .map((principalId) => ({
      principalId,
      sessionCount: sessionsByUser.get(principalId)?.size ?? 0,
      turnCount: turnsByUser.get(principalId) ?? 0,
      lastSeenAt: lastSeenByUser.get(principalId) ?? null,
      admin: adminStatusFromGrants(grants, principalId),
    }))
    .sort((a, b) => Number(b.admin.isAdmin) - Number(a.admin.isAdmin) || (b.lastSeenAt ?? 0) - (a.lastSeenAt ?? 0));
}

export async function reconcileRoster(deps: {
  principals: PrincipalRoster;
  grants: () => Promise<readonly AdminGrant[]>;
  externals: () => Promise<readonly ExternalMember[]>;
}): Promise<number> {
  const now = Date.now();
  return deps.principals.record([
    ...(await deps.grants()).map((g): SeenPerson => ({
      principalId: g.principalId,
      source: "admin",
      at: g.createdAt || now,
    })),
    ...(await deps.externals())
      .filter(isTeammate)
      .map((m): SeenPerson => ({ principalId: m.email, source: "invite", at: m.createdAt })),
  ]);
}
