# Strategy: 4h_WilliamsAlligator_EMA200_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.033 | +19.0% | -8.5% | 273 | FAIL |
| ETHUSDT | 0.198 | +30.0% | -11.4% | 273 | PASS |
| SOLUSDT | 0.396 | +51.9% | -18.7% | 232 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.157 | +7.8% | -8.5% | 98 | PASS |
| SOLUSDT | -0.160 | +3.0% | -10.8% | 86 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with EMA200 trend filter and volume confirmation.
# Williams Alligator uses SMAs of median price: Jaw (13), Teeth (8), Lips (5) with forward shifts.
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > EMA200 AND volume > 1.5x 20-period average
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < EMA200 AND volume > 1.5x 20-period average
# Exit when: Alligator alignment breaks (Lips crosses Teeth) OR price crosses EMA200
# Alligator identifies trend absence/presence; EMA200 filters direction; volume confirms strength.
# Works in trending markets (both bull/bear) by catching sustained moves. Target: 20-30 trades/year per symbol.
name = "4h_WilliamsAlligator_EMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Williams Alligator: SMAs of median price with forward shifts
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values   # Jaw: 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values    # Teeth: 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values     # Lips: 5-period, shifted 3
    
    # EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13+8, 8+5, 5+3, 200, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema200[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        price = close[i]
        ema200_val = ema200[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long entry: Bullish alignment + price > EMA200 + volume spike
            if bullish_alignment and price > ema200_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment + price < EMA200 + volume spike
            elif bearish_alignment and price < ema200_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bullish alignment breaks OR price crosses below EMA200
            if not (lips_val > teeth_val and teeth_val > jaw_val) or price < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bearish alignment breaks OR price crosses above EMA200
            if not (lips_val < teeth_val and teeth_val < jaw_val) or price > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 01:46
