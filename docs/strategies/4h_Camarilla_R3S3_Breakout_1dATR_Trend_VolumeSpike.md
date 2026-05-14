# Strategy: 4h_Camarilla_R3S3_Breakout_1dATR_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.453 | +2.6% | -12.6% | 113 | FAIL |
| ETHUSDT | 0.161 | +27.8% | -16.7% | 103 | PASS |
| SOLUSDT | 0.423 | +49.6% | -14.9% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.284 | +9.4% | -7.7% | 35 | PASS |
| SOLUSDT | -0.081 | +4.5% | -9.4% | 28 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 1d ATR-based trend filter and volume confirmation. Uses HTF 1d for trend alignment (price > 1d close + 0.5*ATR for long, < 1d close - 0.5*ATR for short) to reduce whipsaw. Volume confirmation requires >2.0x 20-bar mean volume. Targets 20-30 trades/year per symbol by requiring strong volume spike and clear trend. Designed to work in both bull (breakouts with volume) and bear (trend-following shorts) markets via disciplined entry/exit.
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
    
    # Get 1d data for HTF trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR(14) on 1d for trend filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: 1d close ± 0.5*ATR
    trend_long = close_1d + 0.5 * atr_1d
    trend_short = close_1d - 0.5 * atr_1d
    
    # Align trend levels to 4h timeframe
    trend_long_aligned = align_htf_to_ltf(prices, df_1d, trend_long)
    trend_short_aligned = align_htf_to_ltf(prices, df_1d, trend_short)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)  # S3 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 1d timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_long_aligned[i]) or 
            np.isnan(trend_short_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend (price > 1d close + 0.5*ATR) with volume confirmation
            # Short: price breaks below Camarilla S3 in downtrend (price < 1d close - 0.5*ATR) with volume confirmation
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > trend_long_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < trend_short_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price moves back below 1d close - 0.5*ATR (trend reversal)
            exit_signal = close[i] < trend_short_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d close + 0.5*ATR (trend reversal)
            exit_signal = close[i] > trend_long_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 19:21
