# Strategy: 6h_elder_ray_12h_ema_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.078 | +13.9% | -15.5% | 232 | DISCARD |
| ETHUSDT | 0.151 | +27.8% | -17.7% | 236 | KEEP |
| SOLUSDT | 1.298 | +292.1% | -28.4% | 195 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.147 | +7.7% | -12.5% | 79 | KEEP |
| SOLUSDT | 0.000 | +4.4% | -14.2% | 75 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h trend filter and volume confirmation
# Long when Bull Power > 0 AND price > 12h EMA(50) AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND price < 12h EMA(50) AND volume > 1.5x 20-period average
# Exit when Bull/Bear Power crosses zero
# Uses Elder Ray (Bull/Bear power) to measure bull/bear strength relative to EMA
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_elder_ray_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(13) for Elder Ray calculation
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # 12h EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    twelve_hour_close = df_12h['close'].values
    twelve_hour_close_series = pd.Series(twelve_hour_close)
    twelve_hour_ema = twelve_hour_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    twelve_hour_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_hour_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(twelve_hour_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Elder Ray power crosses zero
        if position == 1:  # long position
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power > 0 AND price > 12h EMA(50) AND volume confirmation
            if (bull_power[i] > 0 and close[i] > twelve_hour_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price < 12h EMA(50) AND volume confirmation
            elif (bear_power[i] < 0 and close[i] < twelve_hour_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
```

## Last Updated
2026-04-07 04:13
