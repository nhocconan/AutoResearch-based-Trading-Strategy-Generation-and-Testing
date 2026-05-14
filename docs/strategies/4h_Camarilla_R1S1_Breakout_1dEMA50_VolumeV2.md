# Strategy: 4h_Camarilla_R1S1_Breakout_1dEMA50_VolumeV2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.468 | +46.6% | -7.0% | 71 | PASS |
| ETHUSDT | 0.190 | +30.6% | -14.5% | 67 | PASS |
| SOLUSDT | 0.554 | +81.4% | -29.7% | 61 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.132 | -5.7% | -9.3% | 28 | FAIL |
| ETHUSDT | 0.137 | +7.5% | -9.5% | 27 | PASS |
| SOLUSDT | 0.015 | +5.1% | -14.9% | 22 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and daily EMA50 trend filter.
# Uses tighter volume threshold (2.0x EMA20) and EMA50 instead of EMA34 to reduce trades.
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation.
name = "4h_Camarilla_R1S1_Breakout_1dEMA50_VolumeV2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: Range = High - Low
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.0833)
    s1 = close_1d - (range_1d * 1.0833)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    # Align to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_shifted)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_shifted)
    
    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 2.0x 20-period EMA (stricter to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above daily EMA50
            if (price > r1_4h[i] and vol_spike[i] and price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below daily EMA50
            elif (price < s1_4h[i] and vol_spike[i] and price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 (mean reversion to support)
            if price < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R1 (mean reversion to resistance)
            if price > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 00:53
