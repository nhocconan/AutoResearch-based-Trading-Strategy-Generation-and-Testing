# Strategy: 6h_ElderRay_BullBearPower_1dEMA34_VolumeFilter_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.212 | +14.4% | -8.1% | 120 | FAIL |
| ETHUSDT | 0.057 | +22.8% | -9.7% | 106 | PASS |
| SOLUSDT | 1.024 | +123.5% | -16.8% | 91 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.553 | +11.7% | -7.5% | 31 | PASS |
| SOLUSDT | -0.391 | +2.6% | -4.2% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and 6h volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 AND volume > 1.5x 20-bar average.
# Short when Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA34 AND volume > 1.5x 20-bar average.
# Exit when Elder Ray signals weaken (Bull Power <= 0 for longs, Bear Power <= 0 for shorts).
# Uses discrete position size 0.25. Elder Ray measures bull/bear power via EMA, effective in both trending and ranging markets.
# 1d EMA34 ensures we trade with higher timeframe trend. Volume confirms momentum strength.
# Target: 60-120 trades over 4 years (15-30/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume MA (20) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data once before loop for EMA34 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ma_val = vol_ma_20[i]
        ema34_val = ema34_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power becomes non-positive (bullish momentum fading)
            if bull_val <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power becomes non-positive (bearish momentum fading)
            if bear_val <= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 AND volume confirmation
            if bull_val > 0 and bear_val < 0 and price > ema34_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA34 AND volume confirmation
            elif bull_val < 0 and bear_val > 0 and price < ema34_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 07:21
