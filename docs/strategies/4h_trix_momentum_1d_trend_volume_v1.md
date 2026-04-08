# Strategy: 4h_trix_momentum_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.252 | +29.7% | -6.9% | 321 | PASS |
| ETHUSDT | -1.096 | -19.8% | -25.0% | 317 | FAIL |
| SOLUSDT | 0.497 | +54.9% | -19.8% | 303 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.667 | -6.2% | -9.5% | 111 | FAIL |
| SOLUSDT | 0.004 | +5.6% | -10.8% | 102 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_trix_momentum_1d_trend_volume_v1
Hypothesis: TRIX(15) captures momentum on 4h. Long when TRIX > 0 and TRIX > EMA9(TRIX) and price above 1d EMA50 (uptrend). Short when TRIX < 0 and TRIX < EMA9(TRIX) and price below 1d EMA50 (downtrend). Volume confirmation filters weak signals. Works in bull/bear by following higher timeframe trend. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_momentum_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # TRIX(15) on 4h: triple EMA of ROC
    roc = pd.Series(close).pct_change(1)
    ema1 = roc.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values
    
    # Signal line: EMA9 of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False).mean().values
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below signal line or price breaks below EMA50
            if trix[i] < trix_signal[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: TRIX crosses above signal line or price breaks above EMA50
            if trix[i] > trix_signal[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TRIX > 0, TRIX > signal line, with volume and price above EMA50
            if (trix[i] > 0 and trix[i] > trix_signal[i] and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: TRIX < 0, TRIX < signal line, with volume and price below EMA50
            elif (trix[i] < 0 and trix[i] < trix_signal[i] and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 15:26
