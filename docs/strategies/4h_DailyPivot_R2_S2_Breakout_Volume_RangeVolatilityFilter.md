# Strategy: 4h_DailyPivot_R2_S2_Breakout_Volume_RangeVolatilityFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.225 | +33.6% | -17.6% | 66 | PASS |
| ETHUSDT | 0.157 | +28.0% | -24.2% | 65 | PASS |
| SOLUSDT | 0.571 | +100.8% | -41.1% | 54 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.470 | -1.2% | -14.7% | 15 | FAIL |
| ETHUSDT | 0.336 | +12.1% | -16.8% | 25 | PASS |
| SOLUSDT | 1.009 | +33.1% | -17.0% | 13 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily pivot points (standard)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    daily_r2 = daily_pivot + (daily_high - daily_low)
    daily_s2 = daily_pivot - (daily_high - daily_low)
    
    # Align daily levels to 4h timeframe (wait for daily close)
    daily_pivot_4h = align_htf_to_ltf(prices, daily, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, daily, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, daily, daily_s1)
    daily_r2_4h = align_htf_to_ltf(prices, daily, daily_r2)
    daily_s2_4h = align_htf_to_ltf(prices, daily, daily_s2)
    
    # Volume filter: current volume > 2x 20-period average (stricter)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # Range filter: avoid trading near pivot (±1.0%)
    price_to_pivot = np.abs(close - daily_pivot_4h) / daily_pivot_4h
    range_filter = price_to_pivot > 0.01
    
    # Additional filter: avoid trading in low volatility regime (ATR-based)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio = atr / close
    volatility_filter = atr_ratio > 0.01  # Avoid very low volatility periods
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_pivot_4h[i]) or np.isnan(daily_r1_4h[i]) or 
            np.isnan(daily_s1_4h[i]) or np.isnan(daily_r2_4h[i]) or 
            np.isnan(daily_s2_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when all filters pass
        if volume_filter[i] and range_filter[i] and volatility_filter[i]:
            # Long: break above R2 with volume
            if close[i] > daily_r2_4h[i]:
                signals[i] = 0.25
            # Short: break below S2 with volume
            elif close[i] < daily_s2_4h[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyPivot_R2_S2_Breakout_Volume_RangeVolatilityFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-15 08:55
