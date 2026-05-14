# Strategy: 4h_Donchian20_12hEMA50_VolumeFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.257 | +36.2% | -25.3% | 278 | PASS |
| ETHUSDT | 0.286 | +41.2% | -22.8% | 314 | PASS |
| SOLUSDT | 1.061 | +262.4% | -33.9% | 348 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.359 | +0.4% | -7.2% | 101 | FAIL |
| ETHUSDT | 0.709 | +21.6% | -11.2% | 100 | PASS |
| SOLUSDT | 0.731 | +24.2% | -12.0% | 90 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h
    upper_12h = np.full_like(close_12h, np.nan)
    lower_12h = np.full_like(close_12h, np.nan)
    for i in range(19, len(close_12h)):
        upper_12h[i] = np.max(high_12h[i-19:i+1])
        lower_12h[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 50-period EMA on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h volume spike (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above 12h upper Donchian with uptrend and volume spike
            if close[i] > upper_12h_aligned[i] and uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h lower Donchian with downtrend and volume spike
            elif close[i] < lower_12h_aligned[i] and downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12h lower Donchian OR trend reverses
            if (close[i] < lower_12h_aligned[i]) or (not uptrend):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h upper Donchian OR trend reverses
            if (close[i] > upper_12h_aligned[i]) or (not downtrend):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 18:40
