# Strategy: 1h_4h1d_trend_volatility_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.439 | +40.2% | -12.0% | 670 | PASS |
| ETHUSDT | 0.516 | +47.0% | -11.6% | 597 | PASS |
| SOLUSDT | 0.635 | +70.3% | -16.2% | 547 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.796 | +0.6% | -4.7% | 208 | FAIL |
| ETHUSDT | 0.814 | +16.3% | -9.5% | 170 | PASS |
| SOLUSDT | -0.428 | -0.6% | -14.0% | 228 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
1h_4h1d_trend_volatility_breakout_v1
Hypothesis: On 1h timeframe, identify breakout opportunities using 4h trend direction and 1d volatility regime. Go long when price breaks above 4h EMA20 with above-average volume in low volatility regime (favorable for trend continuation). Go short when price breaks below 4h EMA20 with above-average volume in low volatility regime. Use 1d ATR percentile to filter for low volatility environments where breakouts are more likely to succeed. Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_volatility_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile rank (252-day lookback for 1-year)
    atr_percentile = pd.Series(atr_1d).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR percentile below 50th percentile
        low_vol = atr_percentile_aligned[i] < 0.5
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Breakout conditions
        above_ema = close[i] > ema_4h_aligned[i]
        below_ema = close[i] < ema_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses back below EMA or volatility increases significantly
            if close[i] < ema_4h_aligned[i] or atr_percentile_aligned[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses back above EMA or volatility increases significantly
            if close[i] > ema_4h_aligned[i] or atr_percentile_aligned[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if low_vol and vol_ok:
                # Breakout above EMA with volume - go long
                if above_ema and close[i] > ema_4h_aligned[i-1]:
                    position = 1
                    signals[i] = 0.20
                # Breakout below EMA with volume - go short
                elif below_ema and close[i] < ema_4h_aligned[i-1]:
                    position = -1
                    signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-07 19:30
