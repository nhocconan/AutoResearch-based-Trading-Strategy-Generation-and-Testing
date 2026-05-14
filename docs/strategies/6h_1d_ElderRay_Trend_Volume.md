# Strategy: 6h_1d_ElderRay_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.022 | +15.8% | -15.5% | 111 | FAIL |
| ETHUSDT | 0.127 | +25.6% | -20.4% | 107 | PASS |
| SOLUSDT | 1.053 | +240.9% | -33.6% | 104 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.317 | +11.5% | -9.6% | 41 | PASS |
| SOLUSDT | 0.083 | +5.9% | -14.0% | 31 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13, capturing institutional buying/selling pressure.
# Combined with 1d trend (EMA50) and volume spikes, it avoids whipsaws and trades with momentum.
# Works in both bull and bear markets by taking long signals only in uptrend and short in downtrend.
# Target: 12-37 trades per year (50-150 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day EMA(13) for Elder Ray (standard period)
    close_1d = df_1d['close'].values
    ema13_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (13 + 1)
    ema13_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema13_1d[i] = (close_1d[i] - ema13_1d[i-1]) * ema_multiplier + ema13_1d[i-1]
    
    # Calculate 1-day EMA(50) for trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier50 = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier50 + ema50_1d[i-1]
    
    # Calculate daily high and low for Elder Ray
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align all indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate average volume (24-period = 6 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + above daily EMA50 + volume confirmation
            if (bull_power_val > 0 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 (selling pressure) + below daily EMA50 + volume confirmation
            elif (bear_power_val < 0 and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or trend turns down
            if (bull_power_val <= 0 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive or trend turns up
            if (bear_power_val >= 0 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_Trend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 21:41
