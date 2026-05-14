# Strategy: 6h_WilliamsAlligator_1dRegime_Donchian20_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.859 | -13.5% | -18.5% | 52 | FAIL |
| ETHUSDT | 0.152 | +27.6% | -17.4% | 43 | PASS |
| SOLUSDT | 1.134 | +181.2% | -21.0% | 37 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.875 | +22.9% | -8.6% | 16 | PASS |
| SOLUSDT | -0.415 | -2.3% | -13.0% | 14 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Volume Spike Regime
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 1d timeframe to define market regime:
#   - Alligator sleeping (lines intertwined) = range market → mean reversion at extremes
#   - Alligator awakening (lines diverging) = trending market → breakout continuation
# Combines with 6h Donchian(20) breakout for entries and 6h volume > 1.5x 20-period EMA for confirmation
# Designed for 6h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Works in bull markets (breakouts with volume in uptrend regime) and bear markets (breakouts with volume in downtrend regime)
# Williams Alligator provides regime awareness to avoid false breakouts in chop
# Volume spike confirms institutional participation behind breakouts

name = "6h_WilliamsAlligator_1dRegime_Donchian20_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (Smoothed Moving Average - SMMA)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_1d, 13)
    teeth = smma(median_1d, 8)
    lips = smma(median_1d, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 6h data for Donchian channels (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    upper_channel = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_channel)
    
    # Get 6h data for volume EMA
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h volume EMA(20) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ema_20 = pd.Series(vol_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # Alligator regime detection
        # Sleeping (intertwined): max distance between lines < 0.5% of price
        max_line = max(jaw_aligned[i], teeth_aligned[i], lips_aligned[i])
        min_line = min(jaw_aligned[i], teeth_aligned[i], lips_aligned[i])
        alligator_sleeping = (max_line - min_line) / close[i] < 0.005
        
        # Awakening (diverging): lines separated and ordered
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Jaw > Teeth > Lips
        bullish_regime = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_regime = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + bullish regime OR sleeping with bullish bias
            if (close[i] > upper_aligned[i] and volume_confirmed and 
                (bullish_regime or (alligator_sleeping and close[i] > (jaw_aligned[i] + teeth_aligned[i] + lips_aligned[i])/3))):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + bearish regime OR sleeping with bearish bias
            elif (close[i] < lower_aligned[i] and volume_confirmed and 
                  (bearish_regime or (alligator_sleeping and close[i] < (jaw_aligned[i] + teeth_aligned[i] + lips_aligned[i])/3))):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian OR Alligator turns bearish
            if close[i] < lower_aligned[i] or bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Donchian OR Alligator turns bullish
            if close[i] > upper_aligned[i] or bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 03:09
