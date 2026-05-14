# Strategy: 6h_12h_1d_ema_trend_volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.069 | +13.9% | -14.9% | 86 | FAIL |
| ETHUSDT | 0.154 | +28.1% | -17.4% | 80 | PASS |
| SOLUSDT | 1.144 | +244.5% | -30.0% | 71 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.494 | +15.6% | -8.5% | 28 | PASS |
| SOLUSDT | 0.083 | +6.0% | -15.7% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_ema_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume baseline
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume confirmation: 6h volume > 2x 1d average volume (scaled to 6h)
    # 1d volume represents 4x 6h periods, so divide by 4 for per-period comparison
    vol_threshold = vol_avg_1d_aligned / 4.0
    vol_confirm = volume > vol_threshold * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if EMA or volume average not available
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h EMA (trend change)
            if close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h EMA (trend change)
            if close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above 12h EMA with volume confirmation (uptrend)
            if close[i] > ema_50_12h_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below 12h EMA with volume confirmation (downtrend)
            elif close[i] < ema_50_12h_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 09:00
