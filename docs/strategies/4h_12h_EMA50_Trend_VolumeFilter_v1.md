# Strategy: 4h_12h_EMA50_Trend_VolumeFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.050 | +22.7% | -7.8% | 22 | PASS |
| ETHUSDT | -0.490 | -0.0% | -11.8% | 24 | FAIL |
| SOLUSDT | 0.829 | +111.6% | -18.9% | 24 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.013 | +6.2% | -3.2% | 4 | PASS |
| SOLUSDT | -1.982 | -10.1% | -12.5% | 5 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily ATR(34)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(tr_1d)):
        if i < 33:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 33 + tr_1d[i]) / 34
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 12h data for trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_50 = np.full(len(df_12h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_12h)):
        if i < 49:
            ema_12h_50[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema_12h_50[i-1]):
                ema_12h_50[i] = np.mean(close_12h[i-49:i+1])
            else:
                ema_12h_50[i] = close_12h[i] * alpha + ema_12h_50[i-1] * (1 - alpha)
    
    # Align 12h EMA50 to 4h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Calculate ATR(14) for 4h volatility filter
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    atr_4h = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_4h[i] = np.mean(tr_4h[:i+1]) if i > 0 else tr_4h[i]
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 34, 50)  # volume MA needs 20, ATR needs 34, EMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_12h_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # ATR volatility filter: only trade when 4h ATR is above 60% of daily ATR
        vol_filter = atr_4h[i] > atr_1d_aligned[i] * 0.6
        
        if position == 0:
            # Long: price above 12h EMA50 with volume and volatility
            if volume_confirmation and vol_filter and price > ema_12h_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA50 with volume and volatility
            elif volume_confirmation and vol_filter and price < ema_12h_50_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below 12h EMA50 or volatility drops
            if price < ema_12h_50_aligned[i] or atr_4h[i] < atr_1d_aligned[i] * 0.4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above 12h EMA50 or volatility drops
            if price > ema_12h_50_aligned[i] or atr_4h[i] < atr_1d_aligned[i] * 0.4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_12h_EMA50_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 15:17
