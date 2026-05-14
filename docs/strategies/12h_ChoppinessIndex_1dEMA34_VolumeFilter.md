# Strategy: 12h_ChoppinessIndex_1dEMA34_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.316 | +5.0% | -14.2% | 316 | FAIL |
| ETHUSDT | 0.029 | +20.0% | -18.0% | 259 | PASS |
| SOLUSDT | -0.294 | -12.5% | -37.3% | 272 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.483 | +14.9% | -16.9% | 115 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index with 1d EMA trend filter and volume spike.
# Choppiness Index (CHOP) measures market choppiness vs trending.
# CHOP > 61.8 = ranging market (mean reversion opportunity)
# CHOP < 38.2 = trending market (trend following)
# Strategy: In ranging markets (CHOP > 61.8), buy near support (low of day) and sell near resistance (high of day)
# In trending markets (CHOP < 38.2), follow 1d EMA trend with pullback entries
# Volume spike confirms institutional participation in both regimes.
# Designed for ~20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index calculation
    # ATR component
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # High-Low range over period
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hl_range = highest_high - lowest_low
    
    # Avoid division by zero
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(14)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get daily high/low for support/resistance levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop > 61.8 = ranging market (mean reversion)
        # Chop < 38.2 = trending market (trend following)
        if chop[i] > 61.8:  # Ranging market
            # Buy near support (low of day), sell near resistance (high of day)
            if close[i] <= low_1d_aligned[i] * 1.002 and close[i] >= low_1d_aligned[i] * 0.998:  # Near support
                if close[i] > ema34_1d_aligned[i] and volume_filter[i]:  # Only long in uptrend
                    signals[i] = 0.25
                    position = 1
            elif close[i] >= high_1d_aligned[i] * 0.998 and close[i] <= high_1d_aligned[i] * 1.002:  # Near resistance
                if close[i] < ema34_1d_aligned[i] and volume_filter[i]:  # Only short in downtrend
                    signals[i] = -0.25
                    position = -1
        else:  # Trending market (chop < 38.2) or neutral
            # Follow 1d EMA trend with pullback entries
            if close[i] > ema34_1d_aligned[i] and volume_filter[i]:  # Uptrend
                # Buy on pullback to EMA
                if close[i] <= ema34_1d_aligned[i] * 1.01 and close[i] >= ema34_1d_aligned[i] * 0.99:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < ema34_1d_aligned[i] and volume_filter[i]:  # Downtrend
                # Sell on rally to EMA
                if close[i] >= ema34_1d_aligned[i] * 0.99 and close[i] <= ema34_1d_aligned[i] * 1.01:
                    signals[i] = -0.25
                    position = -1
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals

name = "12h_ChoppinessIndex_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-27 19:01
