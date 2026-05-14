# Strategy: 4h_DailyPivot_Volume_ADX_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.223 | +31.9% | -12.0% | 231 | PASS |
| ETHUSDT | 0.254 | +35.6% | -14.3% | 227 | PASS |
| SOLUSDT | 0.069 | +18.7% | -21.2% | 207 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.353 | -7.9% | -12.8% | 84 | FAIL |
| ETHUSDT | 1.271 | +28.5% | -7.9% | 65 | PASS |
| SOLUSDT | -0.457 | -3.3% | -21.4% | 77 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Daily Pivot Point + Volume Spike + ADX Trend Filter
# Uses daily pivot levels (support/resistance) as entry triggers, with volume confirmation
# and ADX trend filter to ensure trades only occur in trending markets
# Pivot points provide objective support/resistance levels that work in both bull and bear markets
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H + L + C) / 3
    # Support 1 = (2*P) - H, Resistance 1 = (2*P) - L
    # Support 2 = P - (H - L), Resistance 2 = P + (H - L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    support_1 = (2 * pivot_point) - high_1d
    resistance_1 = (2 * pivot_point) - low_1d
    support_2 = pivot_point - (high_1d - low_1d)
    resistance_2 = pivot_point + (high_1d - low_1d)
    
    # Align daily pivot levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    s1_aligned = align_htf_to_ltf(prices, df_1d, support_1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, resistance_1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, support_2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, resistance_2)
    
    # ADX calculation (14-period) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(adx[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx[i] < 25:
            # In weak trend/ranging market, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume filter
            if price > r1_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S1 with volume filter
            elif price < s1_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1
            if price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1
            if price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_DailyPivot_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 01:22
