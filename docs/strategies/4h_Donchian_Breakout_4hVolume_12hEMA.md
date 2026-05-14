# Strategy: 4h_Donchian_Breakout_4hVolume_12hEMA

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.149 | +12.7% | -15.9% | 196 | FAIL |
| ETHUSDT | 0.292 | +37.1% | -12.2% | 181 | PASS |
| SOLUSDT | 0.588 | +81.5% | -27.6% | 188 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.475 | +13.0% | -7.1% | 59 | PASS |
| SOLUSDT | 0.204 | +8.7% | -9.6% | 56 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 4h Volume Confirmation and 12h EMA Trend
# Takes long when price breaks above 4h Donchian upper band with volume spike and price > 12h EMA
# Takes short when price breaks below 4h Donchian lower band with volume spike and price < 12h EMA
# Exits when price crosses back below/above the 4h Donchian midline
# Designed to capture strong trends with volume confirmation, avoiding choppy markets
# Target: 20-50 trades per symbol over 4 years (5-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Donchian and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: break above Donchian high with volume spike and bullish trend
            if (price > donchian_high_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                price > ema_12h_aligned[i]):                    # Price above 12h EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low with volume spike and bearish trend
            elif (price < donchian_low_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                  price < ema_12h_aligned[i]):                    # Price below 12h EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian mid
            if price < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian mid
            if price > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_4hVolume_12hEMA"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 02:37
