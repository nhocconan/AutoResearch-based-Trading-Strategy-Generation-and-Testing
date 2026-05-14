# Strategy: 12h_EMA20_Vol_LowVol_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.267 | +3.3% | -15.5% | 87 | FAIL |
| ETHUSDT | 0.677 | +73.8% | -13.3% | 78 | PASS |
| SOLUSDT | 1.576 | +322.6% | -20.3% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.587 | +16.2% | -9.3% | 22 | PASS |
| SOLUSDT | -0.106 | +2.6% | -17.1% | 27 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA20 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 1d True Range for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # Volume filter: require volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Session filter: 00-23 UTC (all hours for 12h timeframe)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 0) & (hour <= 23)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for EMA20 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(atr10_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > EMA20 + volume + low volatility (ATR below median)
            if (close[i] > ema20_1d_aligned[i] and 
                volume_filter[i] and 
                atr10_1d_aligned[i] < np.nanmedian(atr10_1d_aligned[:i+1]) and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < EMA20 + volume + low volatility
            elif (close[i] < ema20_1d_aligned[i] and 
                  volume_filter[i] and 
                  atr10_1d_aligned[i] < np.nanmedian(atr10_1d_aligned[:i+1]) and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < EMA20 (trend change)
            if close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > EMA20 (trend change)
            if close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA20_Vol_LowVol_Filter"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-27 18:02
