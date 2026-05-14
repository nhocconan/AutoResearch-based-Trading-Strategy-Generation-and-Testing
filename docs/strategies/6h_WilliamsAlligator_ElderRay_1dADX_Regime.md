# Strategy: 6h_WilliamsAlligator_ElderRay_1dADX_Regime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.470 | +41.8% | -9.5% | 135 | PASS |
| ETHUSDT | -0.105 | +12.5% | -14.9% | 141 | FAIL |
| SOLUSDT | 0.143 | +27.0% | -24.8% | 101 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.038 | +6.3% | -7.8% | 40 | PASS |
| SOLUSDT | -0.084 | +3.5% | -13.6% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power with 1d regime filter
# Long when: Jaw > Teeth > Lips (bullish alignment) AND Bear Power < 0 (bullish momentum) AND 1d ADX > 25 (trending market)
# Short when: Jaw < Teeth < Lips (bearish alignment) AND Bull Power < 0 (bearish momentum) AND 1d ADX > 25 (trending market)
# Exit when Alligator alignment breaks (Jaw-Teeth-Lips not in proper order)
# Uses 6h timeframe with 1d HTF for ADX regime filter and Elder Ray calculation (target: 50-150 total over 4 years)
# Williams Alligator identifies trend alignment and absence of chop
# Elder Ray measures bull/bear power relative to EMA13
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "6h_WilliamsAlligator_ElderRay_1dADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for ADX and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(np.diff(high_1d))
        tr2 = np.abs(np.diff(low_1d))
        tr3 = np.abs(np.diff(close_1d))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)  # Negative because we want positive values when low decreases
        up_move = np.concatenate([[np.nan], up_move])
        down_move = np.concatenate([[np.nan], down_move])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
            alpha = 1 / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
                else:
                    result[i] = result[i-1]
            return result
        
        tr14 = wilders_smoothing(tr, 14)
        plus_dm14 = wilders_smoothing(plus_dm, 14)
        minus_dm14 = wilders_smoothing(minus_dm, 14)
        
        # Avoid division by zero
        plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
        minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
        
        dx = np.where((plus_di14 + minus_di14) != 0, 
                      np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(len(close_1d), np.nan)
    
    # Calculate 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h Williams Alligator
    # Jaw: Smoothed Median Price (13,8) - Blue line
    # Teeth: Smoothed Median Price (8,5) - Red line  
    # Lips: Smoothed Median Price (5,3) - Green line
    median_price = (high + low) / 2
    
    def alligator_jaw(data, period1=13, period2=8):
        # First smooth with period1, then smooth result with period2
        smoothed1 = pd.Series(data).ewm(span=period1, adjust=False, min_periods=period1).mean().values
        smoothed2 = pd.Series(smoothed1).ewm(span=period2, adjust=False, min_periods=period2).mean().values
        return smoothed2
    
    def alligator_teeth(data, period1=8, period2=5):
        smoothed1 = pd.Series(data).ewm(span=period1, adjust=False, min_periods=period1).mean().values
        smoothed2 = pd.Series(smoothed1).ewm(span=period2, adjust=False, min_periods=period2).mean().values
        return smoothed2
    
    def alligator_lips(data, period1=5, period2=3):
        smoothed1 = pd.Series(data).ewm(span=period1, adjust=False, min_periods=period1).mean().values
        smoothed2 = pd.Series(smoothed1).ewm(span=period2, adjust=False, min_periods=period2).mean().values
        return smoothed2
    
    jaw = alligator_jaw(median_price)
    teeth = alligator_teeth(median_price)
    lips = alligator_lips(median_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Jaw > Teeth > Lips (bullish alignment) AND Bear Power < 0 (bullish momentum) AND ADX > 25
            if (jaw[i] > teeth[i] > lips[i] and 
                bear_power_aligned[i] < 0 and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Jaw < Teeth < Lips (bearish alignment) AND Bull Power < 0 (bearish momentum) AND ADX > 25
            elif (jaw[i] < teeth[i] < lips[i] and 
                  bull_power_aligned[i] < 0 and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (not Jaw > Teeth > Lips)
            if not (jaw[i] > teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (not Jaw < Teeth < Lips)
            if not (jaw[i] < teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 15:47
