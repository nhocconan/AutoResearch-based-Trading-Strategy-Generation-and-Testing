# Strategy: 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirmation_v7

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.239 | +29.7% | -11.6% | 137 | PASS |
| ETHUSDT | 0.444 | +42.0% | -10.5% | 124 | PASS |
| SOLUSDT | 0.542 | +64.2% | -18.4% | 117 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.762 | +0.1% | -5.9% | 55 | FAIL |
| ETHUSDT | 0.779 | +16.0% | -6.3% | 50 | PASS |
| SOLUSDT | 0.270 | +9.2% | -6.5% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirmation_v7
Hypothesis: Further tighten entry by requiring Camarilla breakout + volume spike (>2.0x avg) + 1d EMA34 trend + ATR trailing stop (2.0x). Discrete sizing 0.25. Target 15-25 trades/year to reduce fee drag while maintaining edge. Uses HTF 1d for trend and Camarilla levels to avoid look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) on 4h for breakout confirmation and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Camarilla R1 and S1 from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    if len(high_1d) < 2:
        camarilla_r1 = np.full_like(close_1d_arr, np.nan)
        camarilla_s1 = np.full_like(close_1d_arr, np.nan)
    else:
        camarilla_r1 = close_1d_arr[:-1] + 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_s1 = close_1d_arr[:-1] - 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of volume MA (20), 1d EMA (34), ATR (14)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        # Breakout threshold: price must close beyond Camarilla level by 1.5*ATR (tighter than before)
        breakout_threshold = 1.5 * atr_val
        
        if position == 0:
            # Long: close above R1 + threshold, uptrend (close > EMA34), volume confirmation
            long_signal = (close_val > r1_val + breakout_threshold) and (close_val > ema_34_1d_val) and volume_confirmed
            # Short: close below S1 - threshold, downtrend (close < EMA34), volume confirmation
            short_signal = (close_val < s1_val - breakout_threshold) and (close_val < ema_34_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: price closes below S1
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: trend reversal (close below EMA34)
            elif close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: price closes above R1
            elif close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: trend reversal (close above EMA34)
            elif close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirmation_v7"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 00:39
