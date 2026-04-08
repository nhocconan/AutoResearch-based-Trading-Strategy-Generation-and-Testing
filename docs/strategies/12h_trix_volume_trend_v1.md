# Strategy: 12h_trix_volume_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.034 | +17.8% | -11.5% | 26 | FAIL |
| ETHUSDT | -0.624 | -9.7% | -22.1% | 28 | FAIL |
| SOLUSDT | 0.038 | +19.4% | -27.3% | 23 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.001 | +4.5% | -12.7% | 7 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h_trix_volume_trend_v1
Hypothesis: TRIX (triple-smoothed EMA) on 12h timeframe captures momentum with low lag.
Combined with volume confirmation and daily trend filter (EMA50), it filters false signals.
Works in bull markets via momentum continuation and in bear markets via mean-reversion 
when TRIX diverges from price. Targets 15-35 trades/year by requiring confluence of 
TRIX crossover, volume surge, and daily trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_trix_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=15):
    """Calculate TRIX: triple-smoothed EMA of ROC"""
    # First EMA
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean()
    # Second EMA of first EMA
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    # Third EMA of second EMA
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    # Calculate ROC of triple-smoothed EMA
    trix = ema3.pct_change() * 100
    return trix.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX on 12h data
    trix = calculate_trix(close, 15)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align daily EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # TRIX crossover signals
        trix_cross_up = trix[i-1] < 0 and trix[i] >= 0
        trix_cross_down = trix[i-1] > 0 and trix[i] <= 0
        
        # Daily trend filter
        above_daily_ema50 = close[i] > ema50_1d_aligned[i]
        below_daily_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative or price breaks below daily EMA50
            if trix[i] < 0 or below_daily_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns positive or price breaks above daily EMA50
            if trix[i] > 0 or above_daily_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: TRIX crosses above zero with volume and above daily EMA50
            if trix_cross_up and vol_confirmed and above_daily_ema50:
                position = 1
                signals[i] = 0.25
            # Short: TRIX crosses below zero with volume and below daily EMA50
            elif trix_cross_down and vol_confirmed and below_daily_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 20:33
