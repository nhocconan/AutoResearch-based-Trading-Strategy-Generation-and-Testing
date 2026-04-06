We need to produce a strategy.py for 4h timeframe per instructions. The user says: "You MUST use timeframe = '4h'." Also use HTF = 12h/1d. The hypothesis: We need to state hypothesis in comment at top.

We need to design a strategy that avoids overtrading, uses few conditions, uses price channel breakout or pivot level touch, volume confirmation, regime filter (chop/ADX), ATR stoploss.

We can use Donchian breakout (20) on 4h with volume confirmation and maybe a chop filter (or ADX) from higher timeframe (12h or 1d). Also include ATR stoploss.

We need to keep trade count around 75-200 total over 4 years.

We'll implement:

- Load 4h price data as primary (but we already get prices at 4h). Actually the prices DataFrame is already at the timeframe we set (4h). So we can compute indicators on that directly.

- Load 1d data for chop filter or ADX (or maybe 12h). Use chop to detect ranging vs trending. We'll use chop > 61.8 for range (maybe we avoid trend following in chop? Actually we want to use chop regime filter: if chop > 61.8 (range) we might do mean reversion? But simpler: we only take breakouts when chop < 50 (trending). Many strategies use chop > 61.8 = ranging, chop < 38.2 = trending. We'll use chop < 50 as trending enough to take breakout.

- Compute Donchian channel (20 period high/low) on 4h close.

- Entry long when close crosses above upper Donchian and chop < 50 and volume > volume moving average (maybe 20). Similarly short when close crosses below lower Donchian and chop < 50 and volume > avg.

- Use ATR stoploss: if long and price drops <= entry - 2*ATR, exit; if short and price >= entry + 2*ATR, exit.

- Position size: 0.25.

- Use volume filter: volume > 20-period average * 1.1 (i.e., above average).

- We'll need to compute chop indicator. Choppy index formula: 
   CHOP = 100 * log10( sum(ATR(14)) / (max(high, n) - min(low, n)) ) / log10(n)
   where n = period (usually 14). We'll compute on 1d timeframe.

Implementation steps:

1. Determine n = len(prices). If n < 100, return zeros.

2. Load higher timeframe data: df_1d = get_htf_data(prices, '1d') (once before loop). Compute chop on df_1d.

3. Compute chop array (same length as df_1d). Then align to 4h using align_htf_to_ltf.

4. Compute Donchian upper/lower on 4h: using rolling max/min on high and low (or close?). Usually Donchian uses high and low of lookback period. We'll use high and low.

   upper = rolling max of high over 20 periods.
   lower = rolling min of low over 20 periods.

   We need to ensure we don't use look-ahead: use shifting? Actually for breakout we want to know if current close > previous upper? Typically we break out when close > upper channel of previous period (i.e., using prior 20 bars). So we should shift the upper/lower by 1 to avoid look-ahead. That is, we compute upper = high.rolling(20).max().shift(1). Similarly lower = low.rolling(20).min().shift(1). This way at bar i we use the channel formed from bars i-20 to i-1.

5. Volume filter: compute volume moving average (20) on 4h, then condition volume > ma * 1.1.

6. Chop condition: chop_aligned[i] < 50 (trending). Could also require chop > some threshold for ranging but we just use <50.

7. Entry conditions:
   long: close[i] > upper[i] and chop_aligned[i] < 50 and volume_condition[i]
   short: close[i] < lower[i] and chop_aligned[i] < 50 and volume_condition[i]

8. Exit conditions: 
   If long and (close[i] <= entry_price - 2*ATR[i]) -> exit (set signal 0)
   If short and (close[i] >= entry_price + 2*ATR[i]) -> exit

   Additionally, we could also exit if opposite Donchian break occurs (close < lower for long, close > upper for short) but we can rely on ATR stop.

9. We need to track position and entry price.

10. Use ATR(14) on 4h.

11. Position size: 0.25 (could also use 0.3). We'll stay within 0.25.

12. Ensure we use proper min_periods for rolling.

Let's outline code.

Potential issues: chop calculation may produce NaN for first periods; we need to handle.

Implementation of chop:

Given a DataFrame df with high, low, close.

Compute true range: TR = max(high - low, abs(high - close.prev), abs(low - close.prev))

ATR = rolling mean of TR over period (14).

Then compute sum of ATR over period (14) -> sum_atr = ATR.rolling(window=14).sum()? Actually chop formula uses sum of ATR over n periods (same as ATR period). Usually chop uses ATR(n) and then sum of ATR(n) over n periods? Let's recall:

Choppiness Index (CHOP) = 100 * log10( sum(ATR(n)) / (max(high, n) - min(low, n)) ) / log10(n)

Where sum(ATR(n)) is the sum of ATR values over the last n periods.

Thus we compute ATR(14) then compute rolling sum of ATR over 14 periods.

Then compute max_high = high.rolling(window=14).max()
min_low = low.rolling(window=14).min()
range = max_high - min_low

Then chop = 100 * log10( sum_atr / range ) / log10(14)

If range == 0, we need to handle to avoid division by zero; set chop to 100 maybe.

We'll implement with numpy.

Now, we need to import pandas and numpy.

We'll use the mtf_data helper.

Let's write code.

We'll need to compute indicators before loop for efficiency.

Compute arrays:

- high, low, close, volume from prices.

- Compute upper_donch = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
- lower_donch = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values

- vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
- volume_condition = volume > (vol_ma * 1.1)

- atr: compute TR then ATR(14) using rolling mean.

- chop on 1d: compute ATR14 on 1d, sum_atr14 = rolling sum of ATR14 over 14, max_high1d = rolling max high over 14, min_low1d = rolling min low over 14, range1d = max_high1d - min_low1d, chop1d = 100 * np.log10(sum_atr14 / range1d) / np.log10(14). Need to handle where range1d == 0: set chop to 100 (or maybe 50). We'll avoid division by zero: where range1d == 0, set chop1d = 100.

- Then align chop1d to 4h using align_htf_to_ltf.

Now loop:

Initialize signals array zeros.

position = 0
entry_price = 0.0

Start from lookback: we need enough data for indicators: max(20 (donchian), 20 (vol ma), 14 (atr), chop length maybe 14+14? Actually chop needs 14 for ATR and another 14 for sum, so we need at least 28. We'll start from 100 to be safe.

Loop i from start to n-1:

Check if any needed values are NaN: if any of upper_donch[i], lower_donch[i], vol_ma[i], atr[i], chop_aligned[i] is NaN -> treat as no signal (hold previous? we can set signal to 0 if flat else maintain position? We'll just keep current position signal (i.e., if position !=0, keep same size; else 0). Simpler: if any NaN, set signal to position * size if position !=0 else 0.

But we also need to update position on exit conditions only when we have valid data.

We'll handle inside.

If position == 0:
   Check long condition: close[i] > upper_donch[i] and chop_aligned[i] < 50 and volume_condition[i]
   If true: signal = 0.25; position = 1; entry_price = close[i]
   Else check short condition: close[i] < lower_donch[i] and chop_aligned[i] < 50 and volume_condition[i]
   If true: signal = -0.25; position = -1; entry_price = close[i]
   Else signal = 0

If position == 1 (long):
   Check exit: if close[i] <= entry_price - 2 * atr[i]:
        signal = 0; position = 0
   else:
        signal = 0.25

If position == -1 (short):
   Check exit: if close[i] >= entry_price + 2 * atr[i]:
        signal = 0; position = 0
   else:
        signal = -0.25

That's it.

Now we need to compute chop_aligned.

Potential risk: chop values may be NaN for early bars; we treat as no entry.

Now about timeframe: we set timeframe = "4h".

Now produce final code.

We need to ensure we import numpy as np, pandas as pd, and from mtf_data import get_htf_data, align_htf_to_ltf.

Now produce answer.

</think>

#!/usr/bin/env python3
"""
4h Donchian Breakout with Chop Filter and Volume Confirmation
Hypothesis: Donchian breakouts capture trending moves, while the Choppiness Index (CHOP) filters
out ranging markets to avoid false breakouts. Volume confirmation ensures breakouts have
participation. ATR-based stoploss limits risk. Works in both bull and bear markets by
focusing on strong trending periods identified by low CHOP values (<50). Target: 80-180 total
trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_chop_vol_sl_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Load 1d data for Choppiness Index (once before loop) ---
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Choppiness Index parameters
    chop_period = 14
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # first period
    
    # ATR(14) on 1d
    atr_1d = pd.Series(tr_1d).rolling(window=chop_period, min_periods=chop_period).mean().values
    
    # Sum of ATR over chop_period
    sum_atr_1d = pd.Series(atr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Max high and min low over chop_period
    max_high_1d = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    range_1d = max_high_1d - min_low_1d
    
    # Avoid division by zero: where range == 0, set chop to 100 (max choppy)
    chop_1d = np.full_like(range_1d, 100.0, dtype=np.float64)
    valid_range = range_1d != 0
    if np.any(valid_range):
        chop_1d[valid_range] = 100.0 * np.log10(sum_atr_1d[valid_range] / range_1d[valid_range]) / np.log10(chop_period)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # --- 4h Indicators ---
    # Donchian Channel (20) - use shift(1) to avoid look-ahead
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_condition = volume > (vol_ma * 1.1)  # 10% above average
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Initialize arrays
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after sufficient warmup for all indicators
    start = max(20, 20, 14, chop_period + chop_period)  # donchian, vol, atr, chop
    
    for i in range(start, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop_aligned[i])):
            # Hold current position if any, else flat
            signals[i] = position * 0.25 if position != 0 else 0.0
            continue
        
        if position == 0:  # No position, look for entry
            long_signal = (close[i] > highest_high[i] and
                           chop_aligned[i] < 50.0 and
                           volume_condition[i])
            short_signal = (close[i] < lowest_low[i] and
                            chop_aligned[i] < 50.0 and
                            volume_condition[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: close <= entry - 2*ATR
            if close[i] <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: close >= entry + 2*ATR
            if close[i] >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals