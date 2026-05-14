# Strategy: 6H_ElderRay_BullBearPower_1dEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.341 | +37.5% | -9.3% | 153 | PASS |
| ETHUSDT | 0.200 | +30.9% | -17.2% | 128 | PASS |
| SOLUSDT | 0.775 | +114.0% | -24.2% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.685 | -1.6% | -6.6% | 61 | FAIL |
| ETHUSDT | 0.096 | +6.7% | -8.6% | 48 | PASS |
| SOLUSDT | -0.616 | -5.7% | -18.7% | 48 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume confirmation.
Uses 1d EMA50 to determine trend direction (long only when price > EMA50, short only when price < EMA50).
Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
Enter long when Bull Power > 0 and rising (confirming bullish momentum) in uptrend.
Enter short when Bear Power < 0 and falling (confirming bearish momentum) in downtrend.
Volume spike confirms institutional participation. Designed for 6h timeframe to maintain 12-35 trades/year.
Uses discrete position sizing (0.25) to minimize fee drag while controlling drawdown.
"""

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 6h EMA13 for Elder Ray
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_6h, ema_13)
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_aligned  # Bull Power = High - EMA13
    bear_power = low - ema_13_aligned   # Bear Power = Low - EMA13
    
    # Calculate Elder Ray slope (1-period change) for momentum confirmation
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    # Handle first bar
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND rising AND price above 1d EMA50 (uptrend) AND volume spike
            if bull_power[i] > 0 and bull_power_slope[i] > 0 and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND falling AND price below 1d EMA50 (downtrend) AND volume spike
            elif bear_power[i] < 0 and bear_power_slope[i] < 0 and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray power crosses zero (loss of momentum)
            exit_signal = False
            if position == 1:
                # Exit long when Bull Power <= 0 (bulls losing control)
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short when Bear Power >= 0 (bears losing control)
                if bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 15:29
