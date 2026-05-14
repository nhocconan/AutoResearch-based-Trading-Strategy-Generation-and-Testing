# Strategy: 6h_BollingerBreakout_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.096 | +15.8% | -13.1% | 107 | FAIL |
| ETHUSDT | 0.542 | +51.9% | -9.3% | 94 | PASS |
| SOLUSDT | 0.713 | +95.5% | -26.7% | 91 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.655 | +16.5% | -7.2% | 35 | PASS |
| SOLUSDT | 0.009 | +5.2% | -10.9% | 36 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bands breakout with 12h trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) AND 12h EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below lower BB(20,2) AND 12h EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Bollinger Bands.
# This strategy targets volatility expansion phases with trend alignment to capture momentum moves
# while avoiding choppy markets. The 12h EMA50 filter ensures we trade with the higher timeframe trend.
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 12h trend direction.

name = "6h_BollingerBreakout_12hEMA50_Volume"
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
    
    # Bollinger Bands (20,2)
    bb_length = 20
    bb_mult = 2.0
    sma20 = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    std20 = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = sma20 + (bb_mult * std20)
    lower_band = sma20 - (bb_mult * std20)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 direction
    ema50_rising = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    ema50_falling[1:] = ema50_12h_aligned[1:] < ema50_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_length, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB, 12h EMA50 rising, volume filter
            long_cond = (close[i] > upper_band[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower BB, 12h EMA50 falling, volume filter
            short_cond = (close[i] < lower_band[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Bollinger Bands (below middle band)
            if close[i] < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Bollinger Bands (above middle band)
            if close[i] > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 23:42
