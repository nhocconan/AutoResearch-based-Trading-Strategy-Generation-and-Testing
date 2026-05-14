# Strategy: 4h_ChopRegime_Donchian_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.312 | -3.0% | -25.0% | 69 | FAIL |
| ETHUSDT | 0.089 | +21.6% | -23.6% | 64 | PASS |
| SOLUSDT | 0.485 | +82.2% | -36.5% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.111 | +6.8% | -14.7% | 22 | PASS |
| SOLUSDT | 0.113 | +6.5% | -15.1% | 20 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian breakout
# CHOP > 61.8 = range (mean revert at Donchian bands), CHOP < 38.2 = trend (breakout follow)
# Works in bull/bear: in trend, follow breakouts; in range, fade extremes.
# Uses 1d ATR for Donchian to avoid look-ahead and adapt to volatility.
# Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 10-period Donchian channels (using 1d high/low)
    # Upper band: highest high of last 10 days
    # Lower band: lowest low of last 10 days
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=10, min_periods=10).max().values
    donchian_lower = low_series.rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 14-period ATR for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    atr_series = pd.Series(tr)
    atr = atr_series.rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over last 14 periods
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # True Range for current period (high-low)
    current_tr = high_1d - low_1d
    
    # Choppiness Index: 100 * log(sum(ATR14) / (ATR1 * 14)) / log(100)
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / (current_tr * 14)) / np.log10(100)
    chop = np.where((current_tr > 0) & (atr_sum > 0), chop, 50.0)  # default to neutral
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        
        if chop_val > 61.8:  # Range regime - mean revert at Donchian bands
            # Long: price at or below lower band
            if close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at or above upper band
            elif close[i] >= donchian_upper_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        elif chop_val < 38.2:  # Trend regime - follow breakouts
            # Long: breakout above upper band
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band
            elif close[i] < donchian_lower_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:  # Neutral regime - no trade
            signals[i] = 0.0
            position = 0
    
    return signals

name = "4h_ChopRegime_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 18:32
