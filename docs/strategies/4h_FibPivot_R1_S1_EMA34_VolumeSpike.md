# Strategy: 4h_FibPivot_R1_S1_EMA34_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.221 | +31.3% | -10.7% | 180 | PASS |
| ETHUSDT | 0.240 | +33.6% | -14.5% | 175 | PASS |
| SOLUSDT | 0.685 | +88.4% | -16.8% | 185 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.967 | -4.5% | -9.3% | 64 | FAIL |
| ETHUSDT | 0.477 | +13.7% | -11.3% | 60 | PASS |
| SOLUSDT | 0.410 | +12.6% | -13.7% | 48 | PASS |

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
    
    # === 1d data (HTF for key levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate 1d Fibonacci pivot levels (using previous day's OHLC) ===
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    prev_range = prev_high_1d - prev_low_1d
    
    # Fibonacci-based levels
    r1 = pivot_point + prev_range * 0.382
    s1 = pivot_point - prev_range * 0.382
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation (4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 (stop) or hits R1*1.2 (take profit)
            if price < s1_val or price > r1_val * 1.2:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 (stop) or hits S1*0.8 (take profit)
            if price > r1_val or price < s1_val * 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume AND above 1d EMA34 (uptrend)
            if (price > r1_val) and (price > ema_34_1d_val) and (vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume AND below 1d EMA34 (downtrend)
            elif (price < s1_val) and (price < ema_34_1d_val) and (vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_FibPivot_R1_S1_EMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 18:01
