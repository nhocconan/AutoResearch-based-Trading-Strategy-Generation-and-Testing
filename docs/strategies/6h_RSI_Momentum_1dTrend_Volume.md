# Strategy: 6h_RSI_Momentum_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.051 | +16.8% | -14.7% | 144 | DISCARD |
| ETHUSDT | 0.112 | +25.1% | -12.2% | 141 | KEEP |
| SOLUSDT | 0.699 | +104.9% | -22.9% | 120 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.098 | +6.8% | -7.7% | 43 | KEEP |
| SOLUSDT | -0.512 | -5.0% | -20.8% | 44 | DISCARD |

## Code
```python
# 6h_RSI_Momentum_1dTrend_Volume
# Hypothesis: 6h RSI momentum (RSI(14) > 60 for long, < 40 for short) filtered by 1-day EMA(50) trend
# and volume > 1.5x average. Uses discrete position sizing (±0.25) to limit turnover.
# Works in bull markets via momentum continuation and in bear via mean-reversion off extremes.
# Target: 50-150 total trades over 4 years (~12-37/year).

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 6h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate RSI(14) on 6h data
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rsi = np.full(n, 50.0)  # neutral when undefined
    for i in range(rsi_period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need RSI (14), EMA (50), volume MA (20)
    start_idx = max(rsi_period, ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: RSI > 60 in uptrend with volume
            if rsi[i] > 60 and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: RSI < 40 in downtrend with volume
            elif rsi[i] < 40 and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI < 50 or trend reverses
            if rsi[i] < 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI > 50 or trend reverses
            if rsi[i] > 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_Momentum_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 14:45
