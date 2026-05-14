# Strategy: 6h_ElderRay_1dADXEMA_Regime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.099 | +24.6% | -13.0% | 323 | PASS |
| ETHUSDT | 0.164 | +28.4% | -14.6% | 316 | PASS |
| SOLUSDT | 1.205 | +205.2% | -21.8% | 323 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.393 | -5.8% | -9.7% | 106 | FAIL |
| ETHUSDT | 1.166 | +23.1% | -11.2% | 96 | PASS |
| SOLUSDT | -0.922 | -8.9% | -18.0% | 105 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d Regime Filter.
Long when Bull Power > 0 and Bear Power < 0 with 1d ADX < 20 (range regime) or ADX > 25 with price > EMA20 (trend regime).
Short when Bear Power > 0 and Bull Power < 0 with same regime filters.
Exit when power signals reverse or regime changes.
Uses 1d for ADX/EMA regime, 6h for Elder Ray (EMA13-based Bull/Bear power).
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Get 1d data for regime filters (ADX, EMA)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d EMA20
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        adx_val = adx_14_aligned[i]
        ema20_val = ema20_1d_aligned[i]
        price = close[i]
        
        # Range regime: ADX < 20
        # Trend regime: ADX > 25 and price > EMA20 (for long) or price < EMA20 (for short)
        is_range = adx_val < 20
        is_trend_long = adx_val > 25 and price > ema20_val
        is_trend_short = adx_val > 25 and price < ema20_val
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0
        bear_signal = bear_power[i] > 0
        
        if position == 0:
            # Long: Bull Power positive AND (range regime OR trend regime long)
            if bull_signal and (is_range or is_trend_long):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive AND (range regime OR trend regime short)
            elif bear_signal and (is_range or is_trend_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative OR regime shifts to trend short
            if not bull_signal or is_trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns negative OR regime shifts to trend long
            if not bear_signal or is_trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXEMA_Regime"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 18:53
