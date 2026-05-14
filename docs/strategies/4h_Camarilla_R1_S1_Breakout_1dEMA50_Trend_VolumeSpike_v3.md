# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeSpike_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.139 | +26.9% | -15.2% | 68 | PASS |
| ETHUSDT | -0.539 | -17.7% | -34.5% | 72 | FAIL |
| SOLUSDT | 0.991 | +204.1% | -29.7% | 49 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.144 | +7.5% | -5.4% | 23 | PASS |
| SOLUSDT | -0.225 | -1.2% | -17.7% | 21 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeSpike_v3
Hypothesis: 4h Camarilla R1/S1 breakouts with 1d EMA50 trend filter and volume spike confirmation.
Primary timeframe 4h targets 20-50 trades/year (75-200 total over 4 years) to minimize fee drag.
1d EMA50 provides strong trend alignment that works in both bull and bear markets by filtering counter-trend breakouts.
Volume spike (>2.0x 20-bar average) confirms breakout momentum and reduces false signals.
Designed for BTC/ETH with discrete sizing (0.25) to manage drawdown and avoid overtrading.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1d timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-bar average volume for confirmation on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, volume MA20
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R1 in uptrend (price > 1d EMA50) with volume spike
            # Short: price breaks below Camarilla S1 in downtrend (price < 1d EMA50) with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_1d_aligned[i]) and volume_confirm
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_1d_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below 1d EMA50 (trend reversal)
            exit_signal = close[i] < ema50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal)
            exit_signal = close[i] > ema50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 19:10
