# Strategy: 12h_daily_camarilla_pivot_volume_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.383 | +1.8% | -12.1% | 97 | FAIL |
| ETHUSDT | 0.167 | +28.9% | -10.9% | 86 | PASS |
| SOLUSDT | 0.802 | +127.7% | -20.1% | 71 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.162 | +8.0% | -7.4% | 34 | PASS |
| SOLUSDT | -0.117 | +2.5% | -12.6% | 29 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_v4
# Hypothesis: 12h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation and trend filter.
# Long: Price breaks above Camarilla H3 level, volume > 1.5x 20-period average, price > 50-period SMA.
# Short: Price breaks below Camarilla L3 level, volume > 1.5x 20-period average, price < 50-period SMA.
# Exit: Opposite pivot break or trend reversal (price crosses 50-period SMA).
# Uses 1d Camarilla pivots for structure, volume confirmation to filter weak breakouts, and SMA trend filter.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_v4"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 50-period SMA
    close_s = pd.Series(close)
    sma_50 = close_s.rolling(window=50, min_periods=50).mean().values
    
    # Get 1d data for Camarilla pivots (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from 1d OHLC
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align HTF Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(sma_50[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below L3 OR trend reversal (price < SMA50)
            if low[i] < camarilla_l3_aligned[i] or close[i] < sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above H3 OR trend reversal (price > SMA50)
            if high[i] > camarilla_h3_aligned[i] or close[i] > sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H3, volume confirmed, and uptrend (price > SMA50)
            if (high[i] > camarilla_h3_aligned[i] and volume_confirmed and close[i] > sma_50[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3, volume confirmed, and downtrend (price < SMA50)
            elif (low[i] < camarilla_l3_aligned[i] and volume_confirmed and close[i] < sma_50[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 00:39
