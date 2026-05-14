# Strategy: 6H_VWAP_Deviation_1dEMA50_Trend_ATR_Normalized

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.233 | +12.4% | -8.6% | 88 | FAIL |
| ETHUSDT | 0.050 | +21.7% | -10.5% | 115 | PASS |
| SOLUSDT | -0.329 | -7.0% | -23.7% | 94 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.848 | +17.0% | -5.1% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 1d EMA50 trend filter and ATR-based volatility regime.
In trending markets (price > 1d EMA50 for long, price < 1d EMA50 for short), enter when price deviates significantly from VWAP (mean reversion within trend).
In ranging markets (price near 1d EMA50), avoid trading to reduce whipsaw.
Uses ATR to normalize deviation threshold, adapting to volatility.
Designed for 6h timeframe to achieve 12-35 trades/year with discrete sizing (0.25) to minimize fee drag.
"""

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 6h VWAP (typical price * volume cumsum / volume cumsum)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.full_like(cum_tpv, np.nan), where=cum_vol!=0)
    
    # Calculate 6h ATR(14) for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar TR = high - low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate normalized VWAP deviation: (price - VWAP) / ATR
    vwap_dev = (close - vwap) / atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14)  # need EMA50 and ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below VWAP (oversold) AND uptrend (price > 1d EMA50) AND deviation < -1.0
            if close[i] < vwap[i] and close[i] > ema_50_aligned[i] and vwap_dev[i] < -1.0:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (overbought) AND downtrend (price < 1d EMA50) AND deviation > 1.0
            elif close[i] > vwap[i] and close[i] < ema_50_aligned[i] and vwap_dev[i] > 1.0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP (mean reversion complete) OR trend reversal
            exit_signal = False
            if position == 1:
                # Exit long when price >= VWAP (reversion) OR trend breaks (price <= 1d EMA50)
                if close[i] >= vwap[i] or close[i] <= ema_50_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price <= VWAP (reversion) OR trend breaks (price >= 1d EMA50)
                if close[i] <= vwap[i] or close[i] >= ema_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_VWAP_Deviation_1dEMA50_Trend_ATR_Normalized"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 15:32
