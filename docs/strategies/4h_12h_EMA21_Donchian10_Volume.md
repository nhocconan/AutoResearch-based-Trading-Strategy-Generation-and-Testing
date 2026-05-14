# Strategy: 4h_12h_EMA21_Donchian10_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.687 | -6.8% | -12.3% | 726 | FAIL |
| ETHUSDT | 0.342 | +38.9% | -12.9% | 687 | PASS |
| SOLUSDT | 0.746 | +100.6% | -25.2% | 637 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.036 | +5.8% | -7.3% | 257 | PASS |
| SOLUSDT | -0.990 | -9.5% | -18.5% | 203 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h trend alignment using 12h EMA21 for trend direction,
# 4h Donchian10 breakout for momentum, and volume confirmation. Enters only during 08-20 UTC session.
# Targets 15-30 trades/year (60-120 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends and avoiding choppy markets.
name = "4h_12h_EMA21_Donchian10_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA21 trend (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Get 4h data for Donchian10 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 10-period high/low
    high_10_4h = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    low_10_4h = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    high_10_4h_aligned = align_htf_to_ltf(prices, df_4h, high_10_4h)
    low_10_4h_aligned = align_htf_to_ltf(prices, df_4h, low_10_4h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(high_10_4h_aligned[i]) or 
            np.isnan(low_10_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 12h EMA21 AND breaks 4h Donchian high with volume
            if (close[i] > ema_21_12h_aligned[i] and 
                close[i] > high_10_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA21 AND breaks 4h Donchian low with volume
            elif (close[i] < ema_21_12h_aligned[i] and 
                  close[i] < low_10_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 12h EMA21 or 4h Donchian low
            if close[i] < ema_21_12h_aligned[i] or close[i] < low_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 12h EMA21 or 4h Donchian high
            if close[i] > ema_21_12h_aligned[i] or close[i] > high_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 18:47
