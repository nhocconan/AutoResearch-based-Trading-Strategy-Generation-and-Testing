# Strategy: 4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.138 | +26.7% | -10.7% | 1114 | PASS |
| ETHUSDT | 0.058 | +21.5% | -15.3% | 1106 | PASS |
| SOLUSDT | 0.538 | +76.0% | -24.3% | 1136 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.936 | -4.1% | -10.7% | 349 | FAIL |
| ETHUSDT | 0.370 | +11.8% | -6.8% | 334 | PASS |
| SOLUSDT | 0.479 | +14.1% | -9.9% | 325 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1-day EMA34 trend filter and choppiness regime filter.
Targets 20-40 trades/year by requiring: 1) price breaks daily R1/S1 levels, 2) aligned with 1d EMA34 trend,
3) choppiness index > 61.8 (ranging market) for mean-reversion exits or < 38.2 (trending) for trend continuation.
Uses 4h timeframe to balance trade frequency and capture significant moves. The chop filter avoids false
breakouts in choppy markets and improves performance in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 1d levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                                  np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop_1d = 100 * np.log10(atr_14_1d * 14 / range_14) / np.log10(14) if np.any(range_14 > 0) else 50
    chop_1d = np.where(range_14 > 0, 100 * np.log10(atr_14_1d * 14 / range_14) / np.log10(14), 50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d EMA34 (34) + 1d ATR14 (14) + 1d HH/LL (14)
    start_idx = 34 + 14 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with trend alignment
            # Long breakout: price breaks above R1 with uptrend
            long_breakout = (curr_close > R1_aligned[i]) and uptrend
            # Short breakout: price breaks below S1 with downtrend
            short_breakout = (curr_close < S1_aligned[i]) and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below S1 (mean reversion) or trend changes to downtrend
            # In choppy markets (CHOP > 61.8), exit faster at S1
            # In trending markets (CHOP < 38.2), allow more room
            chop = chop_1d_aligned[i]
            if chop > 61.8:  # Ranging market - mean reversion
                if curr_close < S1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop < 38.2:  # Trending market - trend continuation
                if curr_close < S1_aligned[i] * 0.95 or not uptrend:  # Wider stop in trend
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Transition zone
                if curr_close < S1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above R1 (mean reversion) or trend changes to uptrend
            chop = chop_1d_aligned[i]
            if chop > 61.8:  # Ranging market - mean reversion
                if curr_close > R1_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop < 38.2:  # Trending market - trend continuation
                if curr_close > R1_aligned[i] * 1.05 or not downtrend:  # Wider stop in trend
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Transition zone
                if curr_close > R1_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 11:08
