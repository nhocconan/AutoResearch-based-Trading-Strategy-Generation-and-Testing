# Strategy: 4H_TRIX_VOLUME_SPIKE_CHOPPINESS_REGIME

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.098 | +24.5% | -12.3% | 325 | PASS |
| ETHUSDT | 0.424 | +45.4% | -12.7% | 312 | PASS |
| SOLUSDT | 0.734 | +99.4% | -28.2% | 288 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.324 | +2.7% | -7.1% | 94 | FAIL |
| ETHUSDT | 0.950 | +21.5% | -9.2% | 101 | PASS |
| SOLUSDT | 0.420 | +12.4% | -12.1% | 100 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4H_TRIX_VOLUME_SPIKE_CHOPPINESS_REGIME
Hypothesis: TRIX (triple smoothed EMA) captures momentum shifts, and when combined 
with volume spikes and a chop regime filter (Choppiness Index > 61.8 for mean reversion),
it identifies high-probability reversals in both bull and bear markets. Uses 12h EMA50 
as trend filter to avoid counter-trend trades. Designed for ~25-40 trades/year on 4h 
to minimize fee drag while capturing momentum exhaustion points.
"""
name = "4H_TRIX_VOLUME_SPIKE_CHOPPINESS_REGIME"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX: triple EMA of price, then ROC
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change(1))  # ROC of triple EMA
    trix = trix.fillna(0).values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Choppiness Index: determines if market is ranging (choppy) or trending
    # High CHOP (>61.8) = ranging = good for mean reversion
    # Low CHOP (<38.2) = trending = avoid mean reversion
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    sum_atr = np.sum(atr) if np.sum(atr) > 0 else 1e-10
    chop = 100 * np.log10(highest_high - lowest_low) / np.log10(atr_period) / sum_atr * atr_period
    chop = np.nan_to_num(chop, nan=50.0)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if (np.isnan(trix[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX turning up (momentum shift) + volume spike + choppy market (mean reversion) + above 12h EMA50 (uptrend bias)
            if (trix[i] > trix[i-1] and  # TRIX rising
                volume_spike[i] and 
                chop[i] > 61.8 and  # Choppy/ranging market
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turning down (momentum shift) + volume spike + choppy market + below 12h EMA50 (downtrend bias)
            elif (trix[i] < trix[i-1] and  # TRIX falling
                  volume_spike[i] and 
                  chop[i] > 61.8 and  # Choppy/ranging market
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turning down OR chop drops below 38.2 (trending market)
            if (trix[i] < trix[i-1] or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turning up OR chop drops below 38.2 (trending market)
            if (trix[i] > trix[i-1] or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 10:34
