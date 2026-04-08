# Strategy: 1h_donchian20_1dema50_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.061 | +18.0% | -7.8% | 224 | DISCARD |
| ETHUSDT | -0.609 | -7.5% | -19.1% | 235 | DISCARD |
| SOLUSDT | 0.555 | +70.5% | -25.2% | 220 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.451 | +12.2% | -7.5% | 69 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d filters for directional bias. 
# Uses 4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation (1.5x avg).
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades.
# Target: 60-150 total trades over 4 years (15-37/year) to balance signal quality and fee drag.
# Works in bull via breakouts, in bear via trend filter preventing counter-trend entries.
# 1h provides timely entries while 4h/1d filters ensure only high-probability trades.

name = "1h_donchian20_1dema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period) for breakout signals
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h = align_htf_to_ltf(prices, df_4h, high_20_4h)
    donchian_low_4h = align_htf_to_ltf(prices, df_4h, low_20_4h)
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to 4h Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid_4h[i] or close[i] < donchian_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price returns to 4h Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid_4h[i] or close[i] > donchian_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 4h Donchian breakout + 1d EMA trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high_4h[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above 4h Donchian high with daily uptrend
                    signals[i] = 0.20
                    position = 1
                elif close[i] < donchian_low_4h[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakdown below 4h Donchian low with daily downtrend
                    signals[i] = -0.20
                    position = -1
    
    return signals
```

## Last Updated
2026-04-07 04:13
