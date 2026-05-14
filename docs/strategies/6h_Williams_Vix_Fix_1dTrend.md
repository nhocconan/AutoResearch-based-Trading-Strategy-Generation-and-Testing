# Strategy: 6h_Williams_Vix_Fix_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.289 | +9.7% | -9.7% | 114 | FAIL |
| ETHUSDT | 0.138 | +26.4% | -12.4% | 118 | PASS |
| SOLUSDT | 0.581 | +63.9% | -17.0% | 117 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.846 | +19.5% | -8.2% | 52 | PASS |
| SOLUSDT | 0.303 | +10.4% | -7.2% | 45 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "6h_Williams_Vix_Fix_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams Vix Fix (WVF) - measures market fear
    # High WVF = high fear = potential bottom for mean reversion
    lookback = 22
    highest_high = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
    
    # Avoid division by zero
    wvf = np.full(n, np.nan)
    for i in range(lookback-1, n):
        if highest_high[i] > 0:
            wvf[i] = ((highest_high[i] - low[i]) / highest_high[i]) * 100
    
    # WVF signal: high values indicate fear/oversold
    wvf_ma = np.full(n, np.nan)
    wvf_period = 10
    for i in range(wvf_period-1, n):
        if not np.isnan(wvf[i-wvf_period+1:i+1]).any():
            wvf_ma[i] = np.mean(wvf[i-wvf_period+1:i+1])
    
    # 1d trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(22, n):
        if (np.isnan(wvf_ma[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        wvf_high = wvf_ma[i] > np.percentile(wvf[max(0, i-100):i+1], 80) if i >= 100 else wvf_ma[i] > 50
        
        if position == 0:
            # LONG: High fear (WVF spike) + 1d uptrend + volume confirmation
            if wvf_high and close[i] > ema34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: Low fear (complacency) + 1d downtrend + volume confirmation
            elif not wvf_high and close[i] < ema34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fear subsides or trend breaks
            if wvf_ma[i] < np.percentile(wvf[max(0, i-50):i+1], 50) or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fear increases or trend breaks
            if wvf_ma[i] > np.percentile(wvf[max(0, i-50):i+1], 80) or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 11:20
