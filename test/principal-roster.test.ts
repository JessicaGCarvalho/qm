import { test } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { installPrincipalLinks } from "../src/directory/person.ts";
import { createPrincipalRoster } from "../src/identity/principal-roster.ts";
import { createPostgresPrincipalStore, type PrincipalStore } from "../src/identity/principal-store.ts";
import { testDatabaseUrl } from "./support/test-database.ts";

const url = testDatabaseUrl();
const store = (): PrincipalStore => createPostgresPrincipalStore(url);
const unique = (name: string) => `${name}-${randomUUID().slice(0, 8)}@example.com`;

test("a person joins the roster once, folded to one row, and the first arrival sticks across instances", async () => {
  const pat = unique("pat");
  const backing = store();
  let writes = 0;
  const roster = createPrincipalRoster({
    ...backing,
    fillFirstSeen: (people) => {
      writes++;
      return backing.fillFirstSeen(people);
    },
  });
  await roster.list();
  await roster.note(pat.toUpperCase(), "turn");
  await roster.note(pat, "turn");
  await roster.note(pat, "admin");
  assert.equal(writes, 1, "later notes for someone already on the roster cost no write");
  const row = (await roster.get(pat))!;
  assert.deepEqual([row.principalId, row.source], [pat, "turn"]);
  assert.equal(typeof row.firstSeenAt, "number");

  const other = createPrincipalRoster(store());
  await other.note(pat, "admin");
  assert.equal((await other.get(pat))!.source, "turn", "another instance keeps the first arrival");
});

test("offboarding sets offboarded_at once; reactivation clears it; both reach other instances", async () => {
  const leaver = unique("leaver");
  const writer = createPrincipalRoster(store());
  const reader = createPrincipalRoster(store());
  await reader.list();
  const first = await writer.markOffboarded(leaver, "admin-alice");
  assert.equal(first.offboardedBy, "admin-alice");
  assert.equal(typeof first.offboardedAt, "number");
  const retry = await writer.markOffboarded(leaver.toUpperCase(), "admin-bob");
  assert.deepEqual([retry.offboardedAt, retry.offboardedBy], [first.offboardedAt, "admin-alice"]);

  await reader.note(leaver, "turn");
  const seen = (await createPrincipalRoster(store()).get(leaver))!;
  assert.deepEqual([seen.offboardedBy, seen.source], ["admin-alice", "turn"], "a first arrival keeps the offboarding");

  const back = await reader.markReactivated(leaver);
  assert.deepEqual([back?.offboardedAt, back?.source], [undefined, "turn"]);
  assert.equal((await createPrincipalRoster(store()).get(leaver))!.offboardedAt, undefined);
  const stranger = unique("stranger");
  assert.equal(await writer.markReactivated(stranger), null);
  assert.equal(await writer.get(stranger), null, "reactivating a stranger adds no row");
});

test("recording fills only gaps; linked ids fold to one person with the earliest arrival and offboarding", async () => {
  const sam = unique("sam");
  const slack = `U-${randomUUID().slice(0, 8)}`;
  const left = unique("left");
  const ann = unique("ann");
  installPrincipalLinks({
    canonical: (key) => (key === slack ? sam : undefined),
    aliases: (key) => (key === sam ? [slack] : []),
  });
  try {
    const backing = store();
    const roster = createPrincipalRoster(backing);
    await roster.markOffboarded(left, "admin-alice");
    const people = [
      { principalId: slack, source: "turn" as const, at: 300 },
      { principalId: sam, source: "admin" as const, at: 200 },
      { principalId: left, source: "turn" as const, at: 100 },
      { principalId: ann.toUpperCase(), source: "invite" as const, at: 50 },
    ];
    assert.equal(await roster.record(people), 3);
    assert.deepEqual([(await roster.get(sam))!.source, (await roster.get(sam))!.firstSeenAt], ["admin", 200]);
    assert.deepEqual(
      [(await roster.get(left))!.offboardedBy, (await roster.get(left))!.firstSeenAt],
      ["admin-alice", 100],
    );
    assert.equal((await roster.get(ann))!.principalId, ann);

    await backing.fillFirstSeen([{ principalId: slack, source: "turn", at: 150 }]);
    await backing.markOffboarded(slack, "admin-bob", 10);
    const merged = (await createPrincipalRoster(store()).get(sam))!;
    assert.deepEqual(
      [merged.principalId, merged.firstSeenAt, merged.source, merged.offboardedAt, merged.offboardedBy],
      [sam, 150, "turn", 10, "admin-bob"],
      "a linked id folds into one person: earliest arrival, earliest offboarding",
    );
    assert.equal((await createPrincipalRoster(store()).markReactivated(slack))?.offboardedAt, undefined);

    const again = createPrincipalRoster(store());
    await again.list();
    const late = { principalId: unique("late"), source: "admin" as const, at: 400 };
    assert.equal(await again.record([...people, late]), 1, "a later pass records only the person the roster missed");
    assert.equal(await again.record([...people, late]), 0);
  } finally {
    installPrincipalLinks(null);
  }
});
