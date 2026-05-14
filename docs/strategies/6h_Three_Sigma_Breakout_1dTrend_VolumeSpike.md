# Strategy: 6h_Three_Sigma_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.197 | +9.4% | -15.0% | 171 | FAIL |
| ETHUSDT | 0.707 | +71.8% | -8.6% | 156 | PASS |
| SOLUSDT | 0.972 | +160.9% | -22.8% | 141 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.482 | +13.8% | -7.6% | 48 | PASS |
| SOLUSDT | -0.168 | +1.8% | -12.0% | 51 | FAIL |

## Code
```python
# 6h_Three_Sigma_Breakout_1dTrend_VolumeSpike
# Hypothesis: Three-sigma breakout from 20-period Bollinger Bands on 6h chart with daily trend filter and volume confirmation.
# Uses statistical breakouts (price > MA + 2*std) to capture momentum with tight entry conditions.
# Designed for low trade frequency (<30/year) to minimize fee drag in 2025 bear market.
# Works in both bull (breakouts continue) and bear (mean reversion at extremes) via trend filter.

name = "6h_Three_Sigma_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 20-period Bollinger Bands on 6h
    if n < 20:
        return np.zeros(n)
    
    # Calculate 20-period SMA and std dev
    sma_20 = np.zeros(n)
    std_20 = np.zeros(n)
    
    for i in range(20, n):
        sma_20[i] = np.mean(close[i-20:i])
        std_20[i] = np.std(close[i-20:i])
    
    # Upper and lower bands (2 standard deviations)
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 for Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_20_1d_aligned[i]
        sma = sma_20[i]
        std_val = std_20[i]
        upper = upper_band[i]
        lower = lower_band[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > upper band AND price > 1d EMA20 (uptrend) AND volume > 1.5x average
            if close[i] > upper and close[i] > ema_1d and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < lower band AND price < 1d EMA20 (downtrend) AND volume > 1.5x average
            elif close[i] < lower and close[i] < ema_1d and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < middle band OR trend reverses (price < 1d EMA20)
            if close[i] < sma or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > middle band OR trend reverses (price > 1d EMA20)
            if close[i] > sma or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
```

## Last Updated
2026-05-09 02:59
