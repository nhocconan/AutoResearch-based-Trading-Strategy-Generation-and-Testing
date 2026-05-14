# Strategy: 6h_Adaptive_Keltner_Breakout_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.378 | -1.4% | -21.3% | 45 | FAIL |
| ETHUSDT | 0.174 | +29.5% | -15.2% | 34 | PASS |
| SOLUSDT | 0.660 | +103.5% | -29.4% | 49 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.474 | +14.9% | -11.2% | 12 | PASS |
| SOLUSDT | -0.352 | -3.0% | -16.4% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6h_Adaptive_Keltner_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for Keltner Channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA(34) for trend and midline
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Keltner Channel: midline = EMA(34), upper/lower = EMA(34) ± 2*ATR(14)
    upper_keltner = ema_34 + 2 * atr_1d
    lower_keltner = ema_34 - 2 * atr_1d
    
    # Align to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter using ATR ratio to avoid chop
    high_low_6h = high - low
    high_close_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(high_low_6h, np.maximum(high_close_6h, low_close_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for volume MA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Keltner with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 1.5
            uptrend = close[i] > ema_34_aligned[i]
            
            if close[i] > upper_keltner_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner with volume and in downtrend
            elif close[i] < lower_keltner_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below midline or volatility spike
            if close[i] < ema_34_aligned[i] or atr_6h[i] > np.median(atr_6h[max(0, i-50):i+1]) * 3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above midline or volatility spike
            if close[i] > ema_34_aligned[i] or atr_6h[i] > np.median(atr_6h[max(0, i-50):i+1]) * 3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s adaptive Keltner breakout with trend filter and volume confirmation.
# Uses daily EMA(34) as midline and ATR(14) for dynamic channel width.
# Breaks above/below Keltner channels with volume indicate institutional interest.
# Trend filter ensures trades align with daily direction.
# Volatility filter prevents whipsaws in choppy markets.
# Works in bull (buy upper breaks in uptrend) and bear (sell lower breaks in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~10-25/year.
```

## Last Updated
2026-05-07 14:46
