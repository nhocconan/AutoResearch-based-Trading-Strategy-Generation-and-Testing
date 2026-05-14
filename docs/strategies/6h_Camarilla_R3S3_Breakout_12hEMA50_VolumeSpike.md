# Strategy: 6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.098 | +24.1% | -3.4% | 210 | KEEP |
| ETHUSDT | 0.314 | +32.7% | -6.6% | 185 | KEEP |
| SOLUSDT | 0.338 | +40.2% | -18.3% | 159 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.908 | -6.4% | -7.7% | 83 | DISCARD |
| ETHUSDT | 1.183 | +18.8% | -4.6% | 74 | KEEP |
| SOLUSDT | 0.433 | +10.2% | -5.6% | 57 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts on 6h with 12h EMA50 trend filter and volume spike (>2x 20-bar avg) confirmation.
Trades in direction of 12h trend only. Uses discrete position sizing (0.25) to minimize fee churn.
Designed for low trade frequency (<30/year) to work in both bull and bear markets via trend alignment and volume confirmation.
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
    
    # Get 12h data for HTF trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 12h data (based on previous bar's OHLC)
    camarilla_r3_12h = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3_12h = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    camarilla_h4_12h = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_l4_12h = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align HTF indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4_12h, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4_12h, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume (strict filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend filter and volume spike
            # Long: price breaks above R3 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S3 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-05-06 00:12
