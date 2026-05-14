# Strategy: 6h_1d_Camarilla_R1S1_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.196 | +9.7% | -14.7% | 338 | FAIL |
| ETHUSDT | 0.126 | +26.1% | -12.9% | 312 | PASS |
| SOLUSDT | 1.059 | +176.2% | -28.7% | 263 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.040 | +5.8% | -9.3% | 113 | PASS |
| SOLUSDT | -0.067 | +3.8% | -14.6% | 97 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation from previous day
    # Using close of previous day (shifted by 1)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high_1d, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low_1d, 1)
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: EMA(34) on 6h close
    ema_34 = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or \
           np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_34[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        price_above_ema = price > ema_34[i]
        price_below_ema = price < ema_34[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above EMA34
            if price > r1_6h[i] and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below EMA34
            elif price < s1_6h[i] and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
            # Optional: Strong breakout through R4/S4 for continuation
            elif price > r4_6h[i] and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            elif price < s4_6h[i] and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below S1 (reversal signal) OR breaks below EMA34 (trend change)
            if price < s1_6h[i] or price < ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above R1 (reversal signal) OR breaks above EMA34 (trend change)
            if price > r1_6h[i] or price > ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 10:47
