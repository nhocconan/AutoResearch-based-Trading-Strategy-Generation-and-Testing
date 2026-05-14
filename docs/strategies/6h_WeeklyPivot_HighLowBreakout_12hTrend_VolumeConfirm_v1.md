# Strategy: 6h_WeeklyPivot_HighLowBreakout_12hTrend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.264 | +5.1% | -14.7% | 103 | FAIL |
| ETHUSDT | 0.145 | +27.3% | -12.3% | 91 | PASS |
| SOLUSDT | 0.848 | +144.6% | -29.3% | 80 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.849 | +22.7% | -8.7% | 30 | PASS |
| SOLUSDT | -0.010 | +4.2% | -17.2% | 31 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_WeeklyPivot_HighLowBreakout_12hTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, trade breakouts above/below weekly pivot-derived resistance/support (R1/S1) only when aligned with 12h EMA50 trend and confirmed by volume spike. Weekly pivots provide structure based on prior week's range, reducing noise. 12h EMA50 ensures trend alignment across multiple timeframes. Volume spike confirms momentum. Designed for 6h to capture multi-day moves in both bull and bear markets by filtering with 12h trend. Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (need 5 days for prior week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: using prior week's high, low, close
    # Resample 1d to weekly: Friday's close determines weekly close
    # We need prior week's data, so shift by 5 trading days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get prior week's OHLC (5-day aggregation)
    # For each point, look back 5 days to get prior week's high/low/close
    # Since we're on daily data, we can compute rolling window of 5
    # But we want the completed prior week, so we use values from 5-10 days ago
    # Simpler: use the high/low/close from 5 days ago as weekly proxy
    # More accurate: compute true weekly, but for speed we use 5-day lookback
    # We'll use the high/low/close from 5 periods ago (prior week)
    high_5d_ago = np.roll(high_1d, 5)
    low_5d_ago = np.roll(low_1d, 5)
    close_5d_ago = np.roll(close_1d, 5)
    # For first 5 bars, use first available
    high_5d_ago[:5] = high_1d[0]
    low_5d_ago[:5] = low_1d[0]
    close_5d_ago[:5] = close_1d[0]
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (high_5d_ago + low_5d_ago + close_5d_ago) / 3.0
    weekly_range = high_5d_ago - low_5d_ago
    r1 = 2.0 * weekly_pivot - low_5d_ago
    s1 = 2.0 * weekly_pivot - high_5d_ago
    # For breakout, we use R1 and S1 as key levels
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align all HTF indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR for stoploss calculation (6h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of pivot calc (5), EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(5, 50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        weekly_pivot_val = weekly_pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_12h_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1, above 12h EMA50, with volume spike
            long_signal = (close_val > r1_val) and (close_val > ema_50_val) and vol_spike
            
            # Short: price breaks below S1, below 12h EMA50, with volume spike
            short_signal = (close_val < s1_val) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 OR ATR stoploss (2.5*ATR below entry for wider stop)
            if (close_val < s1_val) or (close_val < entry_price - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR ATR stoploss (2.5*ATR above entry)
            if (close_val > r1_val) or (close_val > entry_price + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_HighLowBreakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-26 03:52
