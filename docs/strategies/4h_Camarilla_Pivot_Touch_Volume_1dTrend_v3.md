# Strategy: 4h_Camarilla_Pivot_Touch_Volume_1dTrend_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.259 | +15.3% | -7.6% | 140 | FAIL |
| ETHUSDT | 0.733 | +47.9% | -7.6% | 104 | PASS |
| SOLUSDT | 0.453 | +46.0% | -9.9% | 91 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.415 | +20.8% | -4.1% | 36 | PASS |
| SOLUSDT | 0.228 | +8.1% | -7.7% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Touch_Volume_1dTrend_v3
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) with volume spike confirmation and 1-day EMA trend filter.
Trades breakouts in trending markets (EMA34) and mean-reversion at pivot levels in ranging markets.
Reduced trade frequency to target 20-30 trades/year by tightening entry conditions: 
- Requires volume spike >2.5x 20-period EMA (increased from 2.0)
- Requires price to be >1.5% away from EMA for breakout entries (reduces whipsaw)
- Adds 2-bar hold minimum after entry to prevent immediate reversals
Designed for low trade frequency to avoid fee drag while capturing high-probability moves.
Works in both bull and bear markets by adapting to trend (breakouts) and range (mean reversion) conditions.
"""

name = "4h_Camarilla_Pivot_Touch_Volume_1dTrend_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Daily Camarilla Pivot Levels (R1, S1) ---
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)  # R1
    s1 = pivot - (range_1d * 1.1 / 12)  # S1
    
    # Align to 4h (Camarilla levels are valid for the entire day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- Volume Spike Detection (2.5x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.5 * vol_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        # Determine trend based on price vs EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        # Distance from EMA as percentage
        ema_distance_pct = abs(close[i] - ema_34_1d_aligned[i]) / ema_34_1d_aligned[i] * 100
        
        # Breakout signals (price crosses R1/S1 with volume spike and sufficient EMA separation)
        # Requires price to be >1.5% away from EMA to avoid whipsaw
        long_breakout = (high[i] > r1_aligned[i]) and vol_spike[i] and (ema_distance_pct > 1.5)
        short_breakout = (low[i] < s1_aligned[i]) and vol_spike[i] and (ema_distance_pct > 1.5)
        
        # Mean reversion at pivot levels (price touches S1/R1 without breakout)
        # Only in non-trending conditions (price near EMA within 0.5%)
        near_ema = ema_distance_pct < 0.5
        long_reversion = (low[i] <= s1_aligned[i]) and near_ema and not vol_spike[i]
        short_reversion = (high[i] >= r1_aligned[i]) and near_ema and not vol_spike[i]
        
        if position == 0:
            # Enforce minimum 2-bar hold after entry (prevents immediate reversal)
            if bars_since_entry < 2:
                signals[i] = 0.0
                continue
                
            if price_above_ema:
                # Uptrend: favor long breakouts, avoid shorts
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
            elif price_below_ema:
                # Downtrend: favor short breakouts, avoid longs
                if short_breakout:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
            else:
                # Near EMA: allow mean reversion at pivot levels
                if long_reversion:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                elif short_reversion:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches S1 (support) or breaks below EMA
                exit_signal = (low[i] <= s1_aligned[i]) or (close[i] < ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R1 (resistance) or breaks above EMA
                exit_signal = (high[i] >= r1_aligned[i]) or (close[i] > ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 05:55
