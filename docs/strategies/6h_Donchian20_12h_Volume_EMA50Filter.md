# Strategy: 6h_Donchian20_12h_Volume_EMA50Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.183 | +10.3% | -15.2% | 93 | FAIL |
| ETHUSDT | 0.136 | +26.7% | -14.5% | 86 | PASS |
| SOLUSDT | 1.176 | +221.8% | -21.0% | 75 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.006 | +5.2% | -10.9% | 28 | PASS |
| SOLUSDT | -0.190 | +1.0% | -17.1% | 27 | FAIL |

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
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === Volume Confirmation (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)  # Moderate volume spike
    
    # === 12h EMA Trend Filter (50-period) ===
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 60  # Need EMA50 and data alignment
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema50 = ema_50_12h_aligned[i]
        
        # === EXIT LOGIC: Exit when price returns to midline (average of Donchian) ===
        midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        
        if position == 1:  # Long position
            # Exit when price crosses back below midline (failed bullish continuation)
            if price < midline:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above midline (failed bearish continuation)
            if price > midline:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation and price > EMA50
            if price > donchian_high_aligned[i] and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian low with volume confirmation and price < EMA50
            elif price < donchian_low_aligned[i] and vol_spike and price < ema50:
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

name = "6h_Donchian20_12h_Volume_EMA50Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 15:32
