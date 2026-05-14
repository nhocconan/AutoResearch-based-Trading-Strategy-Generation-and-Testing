# Strategy: 6h_12hElderRay_BullBearPower_1dEMA34_TrendFilter_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.173 | +28.1% | -10.6% | 381 | PASS |
| ETHUSDT | 0.267 | +34.7% | -10.8% | 400 | PASS |
| SOLUSDT | -0.015 | +12.5% | -31.4% | 405 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.403 | +2.2% | -8.2% | 131 | FAIL |
| ETHUSDT | 0.886 | +20.0% | -6.1% | 103 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray Bull/Bear Power with 1d EMA34 trend filter.
# Long when Bear Power < 0 (bulls in control) AND close > 1d EMA34 (uptrend).
# Short when Bull Power > 0 (bears in control) AND close < 1d EMA34 (downtrend).
# Exit when power signals reverse or price crosses EMA34.
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength relative to EMA.
# 1d EMA34 ensures trading only with higher timeframe trend to avoid whipsaws.
# 6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (capture uptrends) and bear markets (capture downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data once before loop for Elder Ray (EMA13, Bull/Bear Power)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Get 1d data once before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 12h Indicators: Elder Ray (using EMA13) ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # === 1d Indicators: EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # EMA34 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        ema34 = ema34_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Bear Power >= 0 (bulls lose control) OR price < EMA34 (trend break)
            if (bear_power >= 0) or (price < ema34):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Bull Power <= 0 (bears lose control) OR price > EMA34 (trend break)
            if (bull_power <= 0) or (price > ema34):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bear Power < 0 (bulls in control) AND price > EMA34 (uptrend)
            if (bear_power < 0) and (price > ema34):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bull Power > 0 (bears in control) AND price < EMA34 (downtrend)
            elif (bull_power > 0) and (price < ema34):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_12hElderRay_BullBearPower_1dEMA34_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 04:31
