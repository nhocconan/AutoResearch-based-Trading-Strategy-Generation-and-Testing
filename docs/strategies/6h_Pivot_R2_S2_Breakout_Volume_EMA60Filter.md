# Strategy: 6h_Pivot_R2_S2_Breakout_Volume_EMA60Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.027 | +18.0% | -10.0% | 125 | FAIL |
| ETHUSDT | 0.529 | +56.5% | -11.1% | 104 | PASS |
| SOLUSDT | 1.152 | +200.1% | -20.5% | 87 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.215 | +28.6% | -6.8% | 37 | PASS |
| SOLUSDT | -0.297 | -0.6% | -20.1% | 32 | FAIL |

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
    
    # === Daily data for pivot and ATR ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot and R2/S2 levels (using close for pivot)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r2 = pivot + range_hl * 0.618
    s2 = pivot - range_hl * 0.618
    
    # === True Range and ATR (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 6h EMA for trend filter (60-period) ===
    ema_6h = pd.Series(close).ewm(span=60, min_periods=60, adjust=False).mean().values
    
    # Align HTF data to 6h timeframe
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(atr_14_6h[i]) or np.isnan(ema_6h[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r2_level = r2_6h[i]
        s2_level = s2_6h[i]
        atr_val = atr_14_6h[i]
        ema_val = ema_6h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S2 or volatility drops significantly
            if price < s2_level or (i > 0 and atr_val < atr_14_6h[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R2 or volatility drops significantly
            if price > r2_level or (i > 0 and atr_val < atr_14_6h[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R2 with volume spike, above EMA60
            if price > r2_level and vol_spike and price > ema_val:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S2 with volume spike, below EMA60
            elif price < s2_level and vol_spike and price < ema_val:
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

name = "6h_Pivot_R2_S2_Breakout_Volume_EMA60Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 15:58
