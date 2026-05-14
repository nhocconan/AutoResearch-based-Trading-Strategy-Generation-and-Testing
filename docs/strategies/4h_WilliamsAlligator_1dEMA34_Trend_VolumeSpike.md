# Strategy: 4h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.437 | +42.7% | -9.5% | 150 | PASS |
| ETHUSDT | 0.113 | +25.2% | -14.7% | 127 | PASS |
| SOLUSDT | 0.405 | +55.9% | -24.3% | 115 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.595 | -0.3% | -7.9% | 52 | FAIL |
| ETHUSDT | 0.481 | +13.7% | -8.1% | 47 | PASS |
| SOLUSDT | -0.086 | +3.6% | -14.2% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA34 Trend Filter + Volume Spike Confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets.
# Long when Lips > Teeth > Jaw (bullish alignment) with price above Teeth and volume spike.
# Short when Lips < Teeth < Jaw (bearish alignment) with price below Teeth and volume spike.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Designed for 20-50 trades/year on 4h to minimize fee drag while capturing strong trends.
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend.

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 4h data
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().shift(8)
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().shift(5)
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaw_values[i]) or 
            np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND price > Teeth AND 1d uptrend AND volume spike
            if (lips_values[i] > teeth_values[i] and 
                teeth_values[i] > jaw_values[i] and 
                close[i] > teeth_values[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND price < Teeth AND 1d downtrend AND volume spike
            elif (lips_values[i] < teeth_values[i] and 
                  teeth_values[i] < jaw_values[i] and 
                  close[i] < teeth_values[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price closes below Teeth OR 1d trend turns down
            if (lips_values[i] <= teeth_values[i] or 
                teeth_values[i] <= jaw_values[i] or 
                close[i] < teeth_values[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price closes above Teeth OR 1d trend turns up
            if (lips_values[i] >= teeth_values[i] or 
                teeth_values[i] >= jaw_values[i] or 
                close[i] > teeth_values[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-04 15:53
