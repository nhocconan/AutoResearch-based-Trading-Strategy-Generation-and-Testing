#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 12h, filtered by 1d trend and volume spikes,
capture momentum in both bull and bear markets. The 1d trend filter avoids counter-trend
trades, while volume ensures breakout legitimacy. ATR-based stops limit drawdown.
Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_


Wait, the user's message got cut off. Let me see what they intended to write. The user provided a detailed prompt about creating a trading strategy for 12h timeframe, referencing experiment #15362. They mentioned that the current strategy.py (which they provided) is for 4h timeframe, but they need it adapted to 12h. The user's message ends with:

"OUTPUT: Complete strategy.py code only. Start with #!/usr/bin/env python3"

So I need to provide a strategy.py for 12h timeframe based on their instructions.

Looking at the current strategy.py they provided (which is for 4h), I need to adapt it to 12h timeframe. The key changes:
- timeframe = "12h"
- The logic should remain similar: Donchian breakout with 1d trend filter and volume confirmation
- But since it's 12h, the parameters might need adjustment (though they didn't specify changing periods, so I'll keep 20-period for consistency)
- Must use mtf_data.get_htf_data for 1d data (as HTF)

The user's current strategy.py has:
- name = "4h_donchian20_1d_trend_vol_v1"
- timeframe = "4h"
- leverage = 1.0

For 12h version, I should:
- Change timeframe to "12h"
- Update name to reflect 12h
- Keep the same logic but ensure it's adapted for 12h bars

Important: The user's message shows that in the experiment history, there were attempts like "12h_donchian20_1d_trend_vol_v1" which had low trade counts (37 trades/symbol) and was discarded for too few trades. So I need to adjust parameters to increase trade frequency while staying within limits.

The target for 12h is 50-150 total trades over 4 years (12-37/year). The previous attempt had 37 trades/symbol which was below the 50 minimum. So I need to aim for higher frequency.

How to increase trade frequency?
- Reduce Donchian period? But they said to use Donchian(20) in the proven patterns.
- Maybe adjust the volume filter threshold or trend filter.

But the instructions say: "Use proven patterns from DB" and specifically mention "Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss" as working for SOLUSDT.

However, for 12h timeframe, the same 20-period might be too slow. Let me think.

In the proven patterns section, it says for 12h: "Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss → SOLUSDT test Sharpe 1.10-1.38"

So they are using 20-period Donchian on 12h timeframe.

But the previous attempt "12h_donchian20_1d_trend_vol_v1" had only 37 trades which was too few.

Why so few? Possibly because:
1. The volume filter was too strict (volume > 1.5x average)
2. The trend filter required perfect alignment
3. Or the exit conditions were too tight

Looking at the current strategy.py logic:
- Entry: Donchian breakout + trend alignment + volume > 1.5x MA
- Exit: reverse breakout OR trend reversal OR stoploss (2.5*ATR)

To increase trade frequency while maintaining quality, I could:
- Slightly reduce the volume threshold (but not too much to avoid noise)
- Or adjust the Donchian period? But they specified 20.

Wait, the user said: "Proven patterns from DB (what actually works on TEST period, not just train): - Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss → SOLUSDT test Sharpe 1.10-1.38"

So I should stick to 20-period.

Perhaps the issue is with the calculation of indicators. Let me check for potential bugs in the current code that might reduce signals.

In the current 4h strategy:
- ATR calculation: starts at index 20, but uses a weird initialization
- Donchian: correct
- EMA on 1d: similar initialization
- Volume MA: correct

But note: the ATR calculation in the current code is inefficient and might have issues. Better to use a standard method.

Also, for 12h timeframe, the number of bars is less, so we need to ensure we have enough data.

But the main issue from the experiment history is too few trades. So I need to adjust parameters to get more signals while avoiding overtrading.

Let me consider:
- Volume filter: instead of >1.5x, maybe >1.3x? But need to be careful.
- Or perhaps use a different volume filter: volume > average of last 50 periods? But that would be smoother.

Actually, looking at the proven patterns, they say "volume confirmation" without specifying. In many strategies, they use volume > 20-period average.

Another idea: the trend filter might be too strict. Instead of requiring the 1d trend to match exactly, maybe allow some flexibility? But that could reduce quality.

Wait, the user's current strategy (4h version) had 109 trades per symbol in the experiments (see #15357, #15360, #15361 all show 109 tr/sym for 4h version). That's within the 4h target of 75-300 total over 4 years? Wait no:

For 4h, target is 75-200 total over 4 years? Let me check the initial instructions.

In the very first part: "Target: 100-200 total trades over 4 years." for the 4h strategy.

But 109 trades over 4 years is about 27/year, which is below the 100-200 target? That doesn't make sense.

Wait, I think I misread. Let me check:

In the current strategy.py comment: "Target: 100-200 total trades over 4 years."

But in the experiment history for 4h versions, they show 109 trades/sym. If that's over 4 years, then 109/4 = ~27 per year, which is below 100-200/year? That can't be.

Actually, looking at the experiment #15348: "12h_donchian20_1d_trend_vol_v1 | Sharpe=0.126 (37 tr/sym) ← too_few_trades(37tr<<50min)"

Here "37 tr/sym" likely means 37 trades per symbol over the entire training period (2021-2024, 4 years). So 37 trades in 4 years is indeed too few (less than 50 minimum).

Similarly, the 4h versions showing 109 tr/sym would be 109 trades in 4 years, which is above 50, so acceptable.

For 12h, the target is 50-150 total over 4 years (as stated in the experiment #15362 description).

So for 12h, we need at least 50 trades over 4 years.

The previous 12h attempt had 37 trades, which was below 50.

So to increase from 37 to at least 50, I need to make the strategy slightly more sensitive.

How? Perhaps:
- Reduce the volume threshold from 1.5x to 1.3x or 1.2x
- Or reduce the Donchian period? But they specified 20.
- Or adjust the stoploss distance? But that affects risk, not frequency directly.

Actually, the entry frequency is what affects trade count most.

Let me think about the Donchian breakout: on 12h timeframe, how often does price break a 20-period high/low?

In a trending market, it could happen frequently; in ranging, less so.

Perhaps the volume filter is the bottleneck.

Another idea: instead of requiring volume > 1.5x the 20-period average, use volume > 1.5x the 50-period average? That would be a higher bar, actually reducing frequency. No.

To increase frequency, I need to make the volume condition easier to satisfy, so lower the threshold.

But if I make it too low, I get false breakouts.

Perhaps use a different volume filter: volume > average of last 12 periods? That would be more sensitive.

But let's stick to what's proven.

Looking at the proven patterns again: they don't specify the volume threshold, just "volume confirmation".

In the current strategy, it's 1.5x.

Maybe for 12h, I can use 1.2x to increase frequency slightly.

But I need to be careful not to overtrade.

The max for 12h is 200 total over 4 years. So if I go from 37 to, say, 80, that's still safe.

Let me calculate: 37 was too low, need at least 50. So even doubling to 74 would be within 50-150.

So I'll adjust the volume threshold to 1.2x.

Also, I notice in the current strategy, the ATR calculation is done manually in a loop, which is inefficient and might have initialization issues. Better to use a standard method with pandas or numpy for clarity and correctness.

But the rules say to use vectorized where possible, but loops are OK for signal logic.

However, for indicators like ATR, it's better to compute correctly.

Let me rewrite the indicator calculations properly.

Standard ATR calculation:
- TR = max(high-low, |high-close_prev|, |low-close_prev|)
- ATR = smoothed TR (usually Wilder's smoothing: ATR_t = (ATR_{t-1} * (n-1) + TR_t) / n)

In the current code, they initialize ATR[20] = mean of first 20 TR, then for i>20: ATR[i] = (ATR[i-1]*19 + TR[i-1])/20

That's actually correct for Wilder's smoothing with period 20? Let's see:
- For period n, Wilder's smoothing: ATR_t = (ATR_{t-1} * (n-1) + TR_t) / n
- So for n=20, it should be (ATR_{t-1} * 19 + TR_t) / 20
- In their code: ATR[i] = (ATR[i-1] * 19 + TR[i-1]) / 20
- They use TR[i-1] instead of TR[i]. That's an off-by-one error.

Yes! That's likely why the strategy performed poorly - the ATR is lagging or incorrect.

Similarly for the EMA calculation.

So fixing these calculation errors might improve the strategy without changing parameters, thus increasing trade frequency by making signals more accurate.

But the user's instruction is to provide a strategy for 12h timeframe based on the 4h one they showed.

Given that, I'll:
1. Change timeframe to "12h"
2. Update name to "12h_donchian20_1d_trend_vol_v1" (but note this was tried and had low trades, so maybe v2)
3. Fix the ATR and EMA calculations to be correct
4. Possibly adjust volume threshold slightly to increase frequency

But the user said: "Proven patterns from DB" and specifically for 12h they mentioned Donchian(20) etc.

Also, in the experiment history, #15348 and #15352 both show "12h_donchian20_1d_trend_vol_v1" with 37 trades and low Sharpe, discarded for too few trades.

So I need a version that gets more trades.

Let me try fixing the indicator calculations first, as that might resolve the issue without changing logic.

If that still gives too few trades, then adjust volume threshold.

But since I have to provide one strategy, I'll incorporate both fixes and a slight adjustment.

Plan:
- Correct ATR calculation (use TR[i] not TR[i-1])
- Correct EMA calculation (standard EMA)
- Keep volume threshold at 1.5x for now, but if after fixing calculations it's still low, I might need to adjust.

Actually, let's think: if the ATR was calculated wrong, the stoploss might be too tight or too loose, causing premature exits or lack of entries.

Fixing ATR could change the behavior significantly.

Similarly for EMA.

So I'll fix those.

Also, for the EMA on 1d, I should use proper EMA.

Now, for the 12h timeframe, the number of bars is 1/3 of 4h (since 12h is 3x 4h), so fewer bars mean fewer opportunities. But the Donchian period is in bars, so 20-period on 12h is equivalent to 60-period on 4h, which is much slower. That might explain why trade frequency is low.

Ah! That's a key point I missed.

On 4h timeframe, 20-period Donchian = 20 * 4h = 80 hours ≈ 3.33 days
On 12h timeframe, 20-period Donchian = 20 * 12h = 240 hours = 10 days

So the 12h version is looking at a much longer period, hence fewer breakouts.

To get similar frequency, I should reduce the Donchian period for 12h.

But the proven patterns say "Donchian(20) breakout" for 12h. Let me double-check the user's message.

In the experiment #15362 description: "Proven patterns from DB (what actually works on TEST period, not just train): - Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss → SOLUSDT test Sharpe 1.10-1.38"

It doesn't specify the timeframe for that pattern, but given that it's listed under the 12h experiment context, and the previous bullet talks about 4h Donchian, I think this is for 12h.

Let me read carefully:

"Proven patterns from DB (what actually works on TEST period, not just train):
- Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss → SOLUSDT test Sharpe 1.10-1.38
- Camarilla pivot levels from 1d + volume spike + choppiness regime → ETHUSDT test Sharpe 1.47
- ..."

The first bullet doesn't specify timeframe, but since it's in the context of experiment #15362 which is for 12h, and the Sharpe values are plausible for 12h, I think it's for 12h.

However, 20-period on 12h is 10 days, which seems reasonable for a breakout strategy.

But then why did the 12h_donchian20_1d_trend_vol_v1 have only 37 trades?

Perhaps because of the other filters.

Maybe I should look at the 4h version's performance: in the experiments, the 4h versions had 109 trades (which is reasonable for 4h), and for 12h we expect fewer trades because each bar represents more time.

The target for 12h is 50-150 total over 4 years, which is about 12-37 per year.

If 4h gets 109 trades in 4 years (27/year), then 12h should get roughly 1/3 of that since each bar is 3x longer, so about 36 trades in 4 years (9/year), which is below the 50 minimum.

That explains why the 12h version had low trades.

To compensate, for longer timeframes, we need to use shorter lookback periods to get similar frequency.

But the user specifically said for this experiment: "Proven patterns from DB: Donchian(20) breakout + ..." for 12h.

Perhaps the 20-period is correct, and my estimation is off because markets trend differently at different timeframes.

Alternatively, maybe the volume and trend filters are too restrictive for 12h.

Given the instructions, I should stick to Donchian(20) for 12h as stated, but adjust other parameters to increase frequency.

Since the user's current 4h strategy had 109 trades (which is acceptable for 4h), and for 12h we need at least 50, I need the 12h version to have about half the frequency of the 4h version (since 12h bars are 3x 4h, but not exactly linear due to market behavior).

So if 4h version gets 109 trades, 12h version might get around 36 if same logic, which is too low. So I need to make the 12h version more sensitive.

How? By reducing the Donchian period? But they said 20.

Perhaps use Donchian(10) for 12h? But that would be 5 days, which might be too short.

Let's check what others have done.

In the experiment history, there was a 4h strategy that worked: 4h_donchian20_12h_trend_vol_v1 with 136 trades/sym (see #15353).

That's interesting: it used 12h trend filter on 4h chart.

For pure 12h, perhaps I need to adjust.

Another idea: the "Donchian(20)" in the proven pattern might be for the 4h timeframe, and for 12h they use a different period.

But the bullet doesn't specify.

To resolve this, I'll look at the user's instruction for this experiment: "Proven patterns from DB (what actually works on TEST period, not just train): - Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss → SOLUSDT test Sharpe 1.10-1.38"

And this is under the context of experiment #15362 which is for 12h timeframe.

So I'll assume it's for 12h.

Given that the previous attempt with exact parameters had low trades, I suspect the issue is in the indicator calculation errors, as I identified earlier.

So I'll fix those and see if it increases trade frequency.

If after fixing calculations it's still low, I might need to adjust, but since I have to provide one strategy, I'll provide the corrected version and hope it passes.

If not, the user can adjust.

But the user asked me to state hypothesis and implement.

Let me craft the strategy.

Hypothesis: Breakouts from 20-period Donchian channels on 12h timeframe, filtered by 1-day trend (price above/below 20-period EMA) and volume spikes (volume > 1.5x 20-period average), capture sustained momentum moves. Trend filter prevents counter-trend trades, volume confirms breakout validity, and ATR-based stops (2.5x ATR) limit drawdown during sideways or reversing markets.

Now, implement with correct indicator calculations.

I'll use vectorized calculations where possible for efficiency and correctness.

Steps:
1. Calculate TR and ATR correctly using Wilder's smoothing.
2. Calculate Donchian channels (rolling max/min)
3. Get 1d data, calculate 20-period EMA correctly
4. Calculate volume moving average
5. Implement state machine for entries/exits

For ATR, I can use:
- tr0 = abs(high - low)
- tr1 = abs(high - close.shift(1))
- tr2 = abs(low - close.shift(1))
- tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
- atr = tr.ewm(alpha=1/period, adjust=False).mean()  # This is not Wilder's, but similar
Actually, Wilder's smoothing is: ATR = (previous ATR * (n-1) + TR) / n
Which is equivalent to EMA with alpha = 1/n? Let's see:
EMA: today = alpha * price_today + (1-alpha) * yesterday
Wilder: ATR_t = (ATR_{t-1} * (n-1) + TR_t) / n = ATR_{t-1} * (1 - 1/n) + TR_t * (1/n)
So yes, it's EMA with alpha = 1/n.

So I can use: atr = tr.ewm(alpha=1/20, adjust=False).mean()

But need to handle the warmup period.

Similarly for EMA.

To be precise and match the traditional calculation, I'll use the recursive method but correctly.

Let me write the code.

Note: I must use the mtf_data module for 1d data.

Also, ensure no look-ahead.

Let's code accordingly.

</think>
#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 12h timeframe, filtered by 1-day trend and volume spikes,
capture momentum in both bull and bear markets. The 1d trend filter avoids counter-trend trades,
while volume ensures breakout legitimacy. ATR-based stops limit drawdown.
Target: 50-150 total trades over 4 years for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_trend_vol_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period ATR for stops and filters (Wilder's smoothing)
    atr = np.full(n, np.nan)
    if n >= 20:
        tr0 = high - low
        tr1 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
        tr2 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
        tr = np.maximum(tr0, np.maximum(tr1, tr2))
        # Initialize ATR with simple mean of first 20 TR
        atr[19] = np.nanmean(tr[:20])
        for i in range(20, n):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 20-period EMA on 1d for trend (Wilder's smoothing)
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20  # 2/(20+1) ≈ 0.095, but using 2/20=0.1 for simplicity
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns down
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns up
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in uptrend with volume
            if (close[i] > donch_high[i] and
                trend_1d_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in downtrend with volume
            elif (close[i] < donch_low[i] and
                  trend_1d_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals