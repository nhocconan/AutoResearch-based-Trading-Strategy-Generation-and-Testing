# Strategy: 6h_Pivot_R1_S1_Breakout_Volume_EMA34Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.044 | +21.8% | -9.1% | 167 | PASS |
| ETHUSDT | 0.010 | +19.0% | -15.9% | 153 | PASS |
| SOLUSDT | 0.750 | +104.2% | -14.9% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.908 | -12.7% | -16.9% | 68 | FAIL |
| ETHUSDT | 0.806 | +19.7% | -9.7% | 47 | PASS |
| SOLUSDT | -0.565 | -4.5% | -19.1% | 47 | FAIL |

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
    
    # === Daily data for pivot points ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot, R1, S1 levels (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + range_hl * 0.382
    s1 = pivot - range_hl * 0.382
    
    # === 6h EMA for trend filter (34-period) ===
    ema_6h = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align HTF data to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume spike detection (15-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(ema_6h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = r1_6h[i]
        s1_level = s1_6h[i]
        ema_val = ema_6h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S1
            if price < s1_level:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1
            if price > r1_level:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, above EMA34
            if price > r1_level and vol_spike and price > ema_val:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, below EMA34
            elif price < s1_level and vol_spike and price < ema_val:
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

name = "6h_Pivot_R1_S1_Breakout_Volume_EMA34Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 15:59
