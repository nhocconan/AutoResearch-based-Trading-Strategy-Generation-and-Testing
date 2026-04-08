# Strategy: 4h_atr_breakout_1d_trend_volume_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.221 | +31.0% | -10.4% | 27 | PASS |
| ETHUSDT | -0.712 | -8.7% | -30.0% | 24 | FAIL |
| SOLUSDT | 0.787 | +106.5% | -20.2% | 15 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.148 | +4.3% | -5.1% | 11 | FAIL |
| SOLUSDT | 0.228 | +8.9% | -9.9% | 5 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h ATR Breakout + 1d Trend + Volume Confirmation v3
Hypothesis: ATR-based breakouts from volatility-adjusted channels combined with 1d trend filter (EMA 50) and volume confirmation work across bull/bear markets. The 4h timeframe targets 20-50 trades/year, avoiding excessive turnover while capturing significant moves. Volume ensures breakout validity, and 1d trend avoids counter-trend whipsaws. Reduced trade frequency via higher ATR multiplier and volume threshold.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h ATR(14) for volatility and breakout bands
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian-like channels using ATR (wider bands = fewer trades)
    upper_band = np.roll(close, 1) + (2.5 * atr)  # Previous close + 2.5*ATR
    lower_band = np.roll(close, 1) - (2.5 * atr)  # Previous close - 2.5*ATR
    
    # Volume filter (>1.5x 20-period average = stricter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower band or trend reverses
            if close[i] <= lower_band[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper band or trend reverses
            if close[i] >= upper_band[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] >= upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with trend alignment and volume
            elif (close[i] <= lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 00:26
