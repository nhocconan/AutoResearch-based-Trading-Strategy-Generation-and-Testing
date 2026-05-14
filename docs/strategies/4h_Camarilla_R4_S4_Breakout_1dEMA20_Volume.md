# Strategy: 4h_Camarilla_R4_S4_Breakout_1dEMA20_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.042 | +16.5% | -16.1% | 315 | FAIL |
| ETHUSDT | 0.148 | +27.6% | -17.2% | 317 | PASS |
| SOLUSDT | 1.056 | +190.7% | -24.3% | 300 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.167 | +8.1% | -11.6% | 107 | PASS |
| SOLUSDT | 0.330 | +11.4% | -9.8% | 103 | PASS |

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
    
    # Get daily data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter (vectorized with proper initialization)
    close_1d = df_1d['close'].values
    ema_20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        # Initialize with SMA of first 20 values
        ema_20_1d[19] = np.mean(close_1d[:20])
        # Calculate EMA for remaining values
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_20_1d[i-1]
    
    # Calculate previous day's OHLC for Camarilla (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R4 and S4 calculation (wider bands for fewer trades)
    range_hl = prev_high - prev_low
    camarilla_factor = range_hl * 1.1 / 4
    r4 = prev_close + camarilla_factor
    s4 = prev_close - camarilla_factor
    
    # Align daily indicators to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above R4 with volume and above daily EMA20
            if price > r4_aligned[i] and vol_filter and price > ema_20_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below S4 with volume and below daily EMA20
            elif price < s4_aligned[i] and vol_filter and price < ema_20_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below S4 or below daily EMA20
            if price < s4_aligned[i] or price < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above R4 or above daily EMA20
            if price > r4_aligned[i] or price > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_1dEMA20_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 12:51
