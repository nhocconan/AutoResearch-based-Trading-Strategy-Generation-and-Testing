# Strategy: 6h_Ichimoku_Cloud_Breakout_1dTrend_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.095 | +14.5% | -12.2% | 116 | FAIL |
| ETHUSDT | 0.150 | +27.7% | -17.0% | 95 | PASS |
| SOLUSDT | 1.021 | +182.9% | -24.1% | 86 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.027 | +5.5% | -8.5% | 41 | PASS |
| SOLUSDT | -0.713 | -8.8% | -21.4% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_Filter
Hypothesis: Ichimoku cloud on 6h acts as dynamic support/resistance; breaks above/below cloud with volume spike and 1d EMA50 trend filter capture strong momentum moves in both bull and bear markets. Cloud requires price to be fully above/below to avoid whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components."""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past tenkan periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past kijun periods
    highest_kijun = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward kijun periods
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past senkou periods shifted forward kijun periods
    highest_senkou = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    lowest_senkou = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_b = ((highest_senkou + lowest_senkou) / 2)
    
    # Chikou Span (Lagging Span): close shifted back kijun periods (not used for signals)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # 6h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Ichimoku (senkou=52) + EMA (50) + volume MA (20)
    start_idx = max(52, 50, 20) + 5  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: price breaks above cloud with volume spike and 1d uptrend
            long_breakout = (curr_close > cloud_top) and vol_spike[i] and (curr_close > ema_aligned[i])
            # Short: price breaks below cloud with volume spike and 1d downtrend
            short_breakout = (curr_close < cloud_bottom) and vol_spike[i] and (curr_close < ema_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below cloud or trend turns down
            if (curr_close < cloud_bottom) or (curr_close < ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above cloud or trend turns up
            if (curr_close > cloud_top) or (curr_close > ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 11:21
