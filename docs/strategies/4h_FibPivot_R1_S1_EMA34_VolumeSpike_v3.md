# Strategy: 4h_FibPivot_R1_S1_EMA34_VolumeSpike_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.171 | +28.0% | -9.2% | 214 | PASS |
| ETHUSDT | 0.101 | +24.6% | -12.6% | 207 | PASS |
| SOLUSDT | 0.606 | +76.1% | -31.8% | 184 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.013 | -4.4% | -9.7% | 79 | FAIL |
| ETHUSDT | 0.378 | +11.5% | -9.9% | 67 | PASS |
| SOLUSDT | -0.079 | +4.1% | -11.5% | 57 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (HTF for key levels) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Previous Day Range Calculation ===
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = close_12h[0]
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    prev_range = prev_high_12h - prev_low_12h
    
    # === Calculate 12h Pivot Points (Fibonacci-based) ===
    pivot_point = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    
    # Calculate key levels: R1 and S1 at 0.382 Fibonacci
    r1 = pivot_point + prev_range * 0.382
    s1 = pivot_point - prev_range * 0.382
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # === 12h EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === Volume confirmation (4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34_12h_val = ema_34_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 (stop) or hits R1*1.25 (take profit)
            if price < s1_val or price > r1_val * 1.25:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 (stop) or hits S1*0.75 (take profit)
            if price > r1_val or price < s1_val * 0.75:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume AND above 12h EMA34 (uptrend)
            if (price > r1_val) and (price > ema_34_12h_val) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume AND below 12h EMA34 (downtrend)
            elif (price < s1_val) and (price < ema_34_12h_val) and (vol_ratio_val > 2.0):
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

name = "4h_FibPivot_R1_S1_EMA34_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-16 18:03
