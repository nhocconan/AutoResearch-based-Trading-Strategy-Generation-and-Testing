# Strategy: 4h_Bollinger_Breakout_Volume_VolFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.266 | +28.9% | -8.9% | 409 | PASS |
| ETHUSDT | 0.083 | +23.8% | -6.6% | 384 | PASS |
| SOLUSDT | -0.101 | +13.3% | -22.5% | 371 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.848 | +1.6% | -4.3% | 135 | FAIL |
| ETHUSDT | 0.746 | +13.7% | -3.3% | 122 | PASS |

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
    
    # 1-day ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4-hour Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    sma_4h = close_series.rolling(window=20, min_periods=20).mean()
    std_4h = close_series.rolling(window=20, min_periods=20).std()
    upper_4h = sma_4h + 2 * std_4h
    lower_4h = sma_4h - 2 * std_4h
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # ATR-based volatility filter: require ATR > 0.3 * median ATR
    atr_median = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1d_aligned > 0.3 * atr_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i])):
            continue
        
        # Long: close breaks above upper band + volume + volatility filter
        if close[i] > upper_4h[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = 0.25
        
        # Short: close breaks below lower band + volume + volatility filter
        elif close[i] < lower_4h[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = -0.25
        
        # Exit: close crosses back inside bands (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper_4h[i]) or
               (signals[i-1] == -0.25 and close[i] > lower_4h[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Bollinger_Breakout_Volume_VolFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-15 06:45
