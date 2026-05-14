# Strategy: 4H_Camarilla_R1_S1_Breakout_1DTrend_VolumeS_Rev2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.381 | +45.6% | -14.0% | 95 | KEEP |
| ETHUSDT | 0.091 | +22.4% | -18.9% | 104 | KEEP |
| SOLUSDT | 0.789 | +153.0% | -29.3% | 118 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.389 | +0.2% | -7.2% | 42 | DISCARD |
| ETHUSDT | 0.785 | +23.2% | -9.9% | 32 | KEEP |
| SOLUSDT | 0.568 | +18.8% | -9.1% | 32 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1DTrend_VolumeS_Rev2
Hypothesis: Improve the original strategy by adding a momentum filter (RSI) to reduce whipsaws and overtrading.
In bull markets: price above daily EMA34, look for long entries when 4h closes above daily R1 with volume and RSI > 50.
In bear markets: price below daily EMA34, look for short entries when 4h closes below daily S1 with volume and RSI < 50.
This aims to reduce false breakouts and improve win rate while maintaining reasonable trade frequency.
"""
name = "4H_Camarilla_R1_S1_Breakout_1DTrend_VolumeS_Rev2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1D data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + (high - low) * 1.1 / 12, S1 = close - (high - low) * 1.1 / 12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current 4h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # RSI filter (14-period) to avoid whipsaws
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily EMA34, 4h close above daily R1, volume confirmation, and RSI > 50
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > r1_aligned[i] and 
                volume_filter[i] and 
                rsi_values[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34, 4h close below daily S1, volume confirmation, and RSI < 50
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_filter[i] and 
                  rsi_values[i] < 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 08:26
