# Strategy: 6h_WilliamsAlligator_1dElderRay_Trend_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.143 | +16.0% | -18.5% | 79 | FAIL |
| ETHUSDT | 0.069 | +23.2% | -8.3% | 80 | PASS |
| SOLUSDT | 1.061 | +124.9% | -15.8% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.234 | +8.4% | -8.4% | 28 | PASS |
| SOLUSDT | -0.554 | -0.9% | -10.9% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13/8/5) + 1d Elder Ray (Bull/Bear Power) with volume confirmation
# Uses 6h primary timeframe for Alligator trend identification (JAW=13, TEETH=8, LIPS=5)
# 1d Elder Ray confirms trend strength: Bull Power > 0 and Bear Power < 0 for longs, inverse for shorts
# Volume confirmation (1.5x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Alligator provides smooth trend identification, Elder Ray confirms momentum behind the move
# Works in both bull and bear markets by only trading when both indicators agree on direction

name = "6h_WilliamsAlligator_1dElderRay_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    close_1d = pd.Series(df_1d['close'])
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (high_1d - ema13_1d).values
    bear_power = (low_1d - ema13_1d).values
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Williams Alligator on 6h data (JAW=13, TEETH=8, LIPS=5)
    # Alligator lines are SMMA (Smoothed Moving Average) of median price
    median_price = (high + low) / 2
    median_series = pd.Series(median_price)
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = series.rolling(window=period, min_periods=period).mean()
        result = np.full_like(series.values, np.nan, dtype=float)
        result[period-1] = sma.iloc[period-1]
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series.iloc[i]) / period
        return result
    
    jaw = smma(median_series, 13)  # Blue line
    teeth = smma(median_series, 8)  # Red line
    lips = smma(median_series, 5)   # Green line
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and Elder Ray calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator long: Lips > Teeth > Jaw (green > red > blue)
            # Alligator short: Lips < Teeth < Jaw (green < red < blue)
            alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Elder Ray confirmation: Bull Power > 0 and Bear Power < 0 for longs
            # Bear Power < 0 and Bull Power > 0 for shorts (same condition)
            elder_long = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
            elder_short = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
            
            if alligator_long and elder_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif alligator_short and elder_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses (Lips < Teeth) or Elder Ray weakens
            if lips[i] < teeth[i] or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Lips > Teeth) or Elder Ray weakens
            if lips[i] > teeth[i] or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 22:30
