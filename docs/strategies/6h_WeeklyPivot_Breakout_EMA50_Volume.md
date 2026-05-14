# Strategy: 6h_WeeklyPivot_Breakout_EMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.027 | +21.1% | -12.0% | 133 | PASS |
| ETHUSDT | 0.709 | +70.7% | -9.2% | 123 | PASS |
| SOLUSDT | 1.157 | +200.0% | -23.9% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.076 | -4.8% | -8.9% | 49 | FAIL |
| ETHUSDT | 0.877 | +20.6% | -6.5% | 36 | PASS |
| SOLUSDT | 0.558 | +15.0% | -7.4% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data (weekly = 5 daily bars approx)
    # Using 5-period rolling window on daily data
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    # Weekly resistance levels: R1 = 2*P - L, R2 = P + (H - L)
    weekly_r1 = 2 * weekly_pivot - low_5d
    weekly_r2 = weekly_pivot + (high_5d - low_5d)
    # Weekly support levels: S1 = 2*P - H, S2 = P - (H - L)
    weekly_s1 = 2 * weekly_pivot - high_5d
    weekly_s2 = weekly_pivot - (high_5d - low_5d)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.5 * 30-period average
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly pivot, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or 
            np.isnan(ema50_6h[i]) or 
            np.isnan(volume_ma30[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma30[i])
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema50_6h[i]
        price_below_ema = close[i] < ema50_6h[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_6h[i]
        price_below_s1 = close[i] < weekly_s1_6h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above 12h EMA50
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and below 12h EMA50
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot OR below 12h EMA50
            if (close[i] < weekly_pivot_6h[i]) or (close[i] < ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot OR above 12h EMA50
            if (close[i] > weekly_pivot_6h[i]) or (close[i] > ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_EMA50_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 09:58
