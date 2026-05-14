# Strategy: 4h_KAMA_Trend_Filter_Based_on_12h_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.287 | +6.2% | -11.8% | 56 | FAIL |
| ETHUSDT | 0.572 | +56.6% | -11.8% | 51 | PASS |
| SOLUSDT | 0.249 | +35.5% | -22.6% | 48 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.320 | +10.5% | -8.8% | 20 | PASS |
| SOLUSDT | 0.168 | +8.0% | -7.6% | 17 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_Based_on_12h_Trend
Long when KAMA(10) crosses above its 10-period SMA on 4h with 12h EMA50 uptrend and volume > 1.3x average.
Short when KAMA(10) crosses below its 10-period SMA on 4h with 12h EMA50 downtrend and volume > 1.3x average.
Exit when KAMA(10) crosses back through its 10-period SMA.
Uses KAMA for adaptive trend strength, targeting 20-40 trades per year.
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate KAMA on 4h
    kama_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Direction
    diff = np.abs(np.diff(close, prepend=close[0]))
    # Volatility
    volatility = np.abs(np.diff(close))
    volatility_sum = np.zeros(n)
    for i in range(kama_period, n):
        volatility_sum[i] = np.sum(volatility[i - kama_period + 1:i + 1])
    
    # ER (Efficiency Ratio)
    er = np.zeros(n)
    for i in range(kama_period, n):
        if volatility_sum[i] > 0:
            er[i] = diff[i] / volatility_sum[i]
        else:
            er[i] = 0
    
    # SSC (Smoothing Constant)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.full(n, np.nan)
    if n >= kama_period:
        kama[kama_period - 1] = np.mean(close[:kama_period])
        for i in range(kama_period, n):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    # KAMA SMA for crossover
    kama_sma = np.full(n, np.nan)
    sma_period = 10
    for i in range(sma_period - 1, n):
        kama_sma[i] = np.mean(kama[i - sma_period + 1:i + 1])
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA, KAMA SMA, EMA12h, and volume MA20
    start_idx = max(kama_period, sma_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(kama_sma[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.3 * vol_avg
        
        if position == 0:
            # Long: KAMA crosses above its SMA with 12h EMA50 uptrend and volume filter
            if (kama[i] > kama_sma[i] and kama[i - 1] <= kama_sma[i - 1] and 
                price > ema_12h_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: KAMA crosses below its SMA with 12h EMA50 downtrend and volume filter
            elif (kama[i] < kama_sma[i] and kama[i - 1] >= kama_sma[i - 1] and 
                  price < ema_12h_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA crosses below its SMA
            if kama[i] < kama_sma[i] and kama[i - 1] >= kama_sma[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA crosses above its SMA
            if kama[i] > kama_sma[i] and kama[i - 1] <= kama_sma[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_Filter_Based_on_12h_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 11:17
