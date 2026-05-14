# Strategy: 4h_RSI_Trend_Filter_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.218 | +8.4% | -12.7% | 222 | FAIL |
| ETHUSDT | 0.025 | +19.1% | -15.7% | 220 | PASS |
| SOLUSDT | 0.890 | +138.1% | -24.1% | 201 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.101 | +6.9% | -9.6% | 76 | PASS |
| SOLUSDT | 0.230 | +9.3% | -12.4% | 75 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_RSI_Trend_Filter_Volume_v1
Hypothesis: On 4h timeframe, use RSI(14) with 1d EMA trend filter and volume confirmation to capture momentum in trending markets. Long when RSI>55 and price above 1d EMA; short when RSI<45 and price below 1d EMA. Volume must be above 1.5x average to confirm strength. This avoids overtrading by requiring three clear conditions and focuses on high-probability moves in both bull and bear markets.
"""
name = "4h_RSI_Trend_Filter_Volume_v1"
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
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # RSI(14) calculation on price series
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades (24 hours on 4h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Long: RSI > 55 + price above 1d EMA + volume filter
            if (rsi_values[i] > 55 and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: RSI < 45 + price below 1d EMA + volume filter
            elif (rsi_values[i] < 45 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: RSI returns to neutral zone (45-55) or trend reversal
            if position == 1 and (rsi_values[i] < 50 or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (rsi_values[i] > 50 or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-05-07 07:35
