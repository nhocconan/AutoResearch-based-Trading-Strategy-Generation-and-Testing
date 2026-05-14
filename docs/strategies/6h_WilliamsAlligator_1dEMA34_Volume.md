# Strategy: 6h_WilliamsAlligator_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.081 | +18.7% | -7.7% | 149 | FAIL |
| ETHUSDT | 0.234 | +29.9% | -6.6% | 153 | PASS |
| SOLUSDT | 1.085 | +121.2% | -11.1% | 143 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.240 | +8.2% | -6.2% | 48 | PASS |
| SOLUSDT | 0.114 | +7.0% | -6.8% | 46 | PASS |

## Code
```python
# #!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# Long when Lips > Teeth > Jaw (bullish alignment) and price above Lips
# Short when Lips < Teeth < Jaw (bearish alignment) and price below Lips
# Filtered by 1d EMA34 trend (must align with Alligator signal)
# Volume confirmation: current volume > 1.3x 20-period average
# Williams Alligator uses smoothed moving averages (SMMA) which reduces whipsaw
# Effective in both trending and ranging markets due to alignment requirement
# Targets 50-150 total trades over 4 years (12-37/year) for optimal fee drag

name = "6h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (SMMA)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple moving average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator components
    jaw = smma(close, 13)   # Jaw (13-period)
    teeth = smma(close, 8)  # Teeth (8-period)
    lips = smma(close, 5)   # Lips (5-period)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        # Williams Alligator signals
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        price_above_lips = close_val > lips_val
        price_below_lips = close_val < lips_val
        
        if position == 0:
            # Enter long: bullish alignment, price above lips, 1d uptrend, volume confirmation
            if bullish_alignment and price_above_lips and ema34_1d_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment, price below lips, 1d downtrend, volume confirmation
            elif bearish_alignment and price_below_lips and ema34_1d_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment or price below lips or 1d trend down
            if bearish_alignment or not price_above_lips or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment or price above lips or 1d trend up
            if bullish_alignment or not price_below_lips or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 14:30
