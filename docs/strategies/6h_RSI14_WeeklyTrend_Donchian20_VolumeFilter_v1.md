# Strategy: 6h_RSI14_WeeklyTrend_Donchian20_VolumeFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.235 | +12.7% | -11.9% | 28 | FAIL |
| ETHUSDT | 0.062 | +22.8% | -16.7% | 20 | PASS |
| SOLUSDT | 0.551 | +65.7% | -18.2% | 25 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.682 | +14.7% | -7.5% | 11 | PASS |
| SOLUSDT | -0.274 | +1.7% | -12.4% | 9 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14) for trend filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(1, len(gain)):
        if i < 14:
            avg_gain[i] = (avg_gain[i-1] * i + gain[i]) / (i + 1)
            avg_loss[i] = (avg_loss[i-1] * i + loss[i]) / (i + 1)
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(avg_loss == 0, 100, rsi_1w)
    rsi_1w = np.where(avg_gain == 0, 0, rsi_1w)
    
    # Align weekly RSI to 6h
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 14-period ATR for volatility
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 20-period high/low for Donchian breakout
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 20
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Weekly RSI thresholds: >60 = bullish bias, <40 = bearish bias
        if position == 0:
            # Long: Price breaks above Donchian high with volume and weekly bullish bias
            if price > high_max[i] and vol_ratio > 2.0 and rsi_1w_aligned[i] > 60:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and weekly bearish bias
            elif price < low_min[i] and vol_ratio > 2.0 and rsi_1w_aligned[i] < 40:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or weekly RSI turns bearish
            if price < low_min[i] or rsi_1w_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or weekly RSI turns bullish
            if price > high_max[i] or rsi_1w_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI14_WeeklyTrend_Donchian20_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 12:16
