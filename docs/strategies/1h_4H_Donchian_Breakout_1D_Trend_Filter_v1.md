# Strategy: 1h_4H_Donchian_Breakout_1D_Trend_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.245 | +9.2% | -15.5% | 138 | FAIL |
| ETHUSDT | 0.063 | +22.3% | -12.3% | 135 | PASS |
| SOLUSDT | 0.737 | +102.8% | -26.0% | 144 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.028 | +5.9% | -6.8% | 43 | PASS |
| SOLUSDT | -0.589 | -4.4% | -15.5% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
1h_4H_Donchian_Breakout_1D_Trend_Filter_v1
Hypothesis: Use 4h Donchian(20) breakout for signal direction and 1d EMA(34) for trend filter. 
Long when price breaks above 4h Donchian upper band and close > 1d EMA(34); 
Short when price breaks below 4h Donchian lower band and close < 1d EMA(34).
Volume confirmation: current volume > 1.5x 20-period average volume.
Session filter: only trade between 08:00-20:00 UTC.
This combines trend-following structure with volume confirmation and session filtering to reduce false signals and manage trade frequency.
"""
name = "1h_4H_Donchian_Breakout_1D_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) - upper and lower bands
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 12 bars between trades (12 hours on 1h TF) to reduce frequency
            if bars_since_exit < 12:
                continue
                
            # Check session filter
            if not session_filter[i]:
                continue
                
            # Long: price breaks above 4h Donchian upper + close > 1d EMA + volume filter
            if (close[i] > donch_high_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_exit = 0
            # Short: price breaks below 4h Donchian lower + close < 1d EMA + volume filter
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian band or trend reversal
            if position == 1 and (close[i] < donch_low_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (close[i] > donch_high_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals
```

## Last Updated
2026-05-07 07:36
