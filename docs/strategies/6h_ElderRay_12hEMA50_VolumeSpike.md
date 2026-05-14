# Strategy: 6h_ElderRay_12hEMA50_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.067 | +15.7% | -11.7% | 199 | DISCARD |
| ETHUSDT | 0.116 | +25.3% | -14.7% | 177 | KEEP |
| SOLUSDT | 1.274 | +241.8% | -22.1% | 139 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.026 | +5.4% | -10.2% | 60 | KEEP |
| SOLUSDT | -0.235 | +0.4% | -13.6% | 57 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA(50) trend filter and volume confirmation
# Elder Ray measures bullish/bearish power: Bull Power = High - EMA, Bear Power = Low - EMA
# Trend filter: 12h EMA(50) ensures alignment with higher timeframe trend
# Volume confirmation: current volume > 1.8x 20-period EMA reduces false signals
# Works in bull/bear markets by following 12h trend direction
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag

name = "6h_ElderRay_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h EMA(13) for Elder Ray (standard period)
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Elder Ray signals with 12h trend filter
        # Long: Bull Power > 0 (bulls in control) + price above 12h EMA50 + volume spike
        # Short: Bear Power < 0 (bears in control) + price below 12h EMA50 + volume spike
        if position == 0:
            if bull_power[i] > 0 and close[i] > ema_50_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif bear_power[i] < 0 and close[i] < ema_50_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bulls lose control OR price breaks below 12h EMA50
            if bull_power[i] <= 0 or close[i] <= ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bears lose control OR price breaks above 12h EMA50
            if bear_power[i] >= 0 or close[i] >= ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 12:03
