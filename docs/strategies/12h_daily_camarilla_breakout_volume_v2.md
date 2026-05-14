# Strategy: 12h_daily_camarilla_breakout_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.380 | +6.5% | -20.6% | 179 | FAIL |
| ETHUSDT | 0.035 | +21.5% | -20.6% | 158 | PASS |
| SOLUSDT | -0.260 | -1.9% | -36.8% | 141 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.180 | +8.0% | -6.8% | 60 | PASS |

## Code
```python
#!/usr/bin/env python3
# 12h_daily_camarilla_breakout_volume_v2
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and ATR filter.
# Long: Price breaks above daily R4 level with volume > 1.5x 20-period average AND ATR(14) > 0.5 * ATR(50) (volatility filter).
# Short: Price breaks below daily S4 level with volume > 1.5x 20-period average AND ATR(14) > 0.5 * ATR(50).
# Exit: Price returns to daily pivot point (PP) or breaks opposite S4/R4 level.
# Uses daily Camarilla for key support/resistance, 12h for execution, volume and volatility for confirmation.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_breakout_volume_v2"
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
    
    # ATR for volatility filter
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr.rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.5 * atr_50)  # Only trade when volatility is above half of long-term average
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_1d * 1.1 / 2.0
    s4 = close_1d - range_1d * 1.1 / 2.0
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Volatility filter: only trade when short-term ATR > 50% of long-term ATR
        vol_filter = volatility_filter[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to daily pivot or breaks below S4
            if close[i] <= pivot_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to daily pivot or breaks above R4
            if close[i] >= pivot_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and volatility filter
            bullish_breakout = (close[i] > r4_aligned[i]) and volume_confirmed and vol_filter
            bearish_breakout = (close[i] < s4_aligned[i]) and volume_confirmed and vol_filter
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 00:53
