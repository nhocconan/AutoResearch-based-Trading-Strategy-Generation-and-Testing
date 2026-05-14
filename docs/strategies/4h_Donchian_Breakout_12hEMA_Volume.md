# Strategy: 4h_Donchian_Breakout_12hEMA_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.028 | +20.3% | -18.2% | 122 | PASS |
| ETHUSDT | 0.206 | +32.2% | -13.7% | 112 | PASS |
| SOLUSDT | 0.906 | +160.5% | -25.8% | 114 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.430 | -9.3% | -11.0% | 52 | FAIL |
| ETHUSDT | 0.132 | +7.4% | -9.8% | 41 | PASS |
| SOLUSDT | 0.562 | +16.8% | -12.2% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h Trend Filter and Volume Confirmation
# Uses Donchian channel breakout (20-period) from 4h for entry signals
# 12h EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.8x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of 12h trend
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA (50) for trend direction
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channel (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and Donchian calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 12h EMA
        above_ema = price > ema_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter and above 12h EMA
            if price > donchian_high[i] and vol > 1.8 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter and below 12h EMA
            elif price < donchian_low[i] and vol > 1.8 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low (reversal) or below 12h EMA
            if price < donchian_low[i] or price < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high (reversal) or above 12h EMA
            if price > donchian_high[i] or price > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 01:34
