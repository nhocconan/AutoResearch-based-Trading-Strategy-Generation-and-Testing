# Strategy: 4h_CHOP_Range_EMA34_Bias_Volume_Session

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.220 | +30.2% | -17.5% | 80 | PASS |
| ETHUSDT | 0.353 | +39.6% | -12.0% | 91 | PASS |
| SOLUSDT | -0.075 | +9.6% | -31.4% | 109 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.272 | -6.2% | -9.5% | 37 | FAIL |
| ETHUSDT | 0.364 | +11.4% | -10.0% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) - range detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Sum of True Range over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Higher CHOP = more ranging, Lower CHOP = more trending
    chop_raw = 100 * np.log10(atr14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop = pd.Series(chop_raw).fillna(50).values  # Fill NaN with neutral 50
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: require volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Session filter: 08-20 UTC (active trading hours)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 34  # need 34 for EMA34 and CHOP
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CHOP > 55 (ranging) + price > EMA34 (bullish bias) + volume + session
            if (chop_aligned[i] > 55 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: CHOP > 55 (ranging) + price < EMA34 (bearish bias) + volume + session
            elif (chop_aligned[i] > 55 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: CHOP < 40 (trending) OR price < EMA34 (trend change)
            if (chop_aligned[i] < 40 or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CHOP < 40 (trending) OR price > EMA34 (trend change)
            if (chop_aligned[i] < 40 or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CHOP_Range_EMA34_Bias_Volume_Session"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 17:55
