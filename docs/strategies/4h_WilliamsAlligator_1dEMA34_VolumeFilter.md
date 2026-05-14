# Strategy: 4h_WilliamsAlligator_1dEMA34_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.117 | +25.3% | -18.2% | 64 | PASS |
| ETHUSDT | 0.046 | +17.7% | -24.3% | 64 | PASS |
| SOLUSDT | 0.826 | +170.8% | -33.6% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.294 | +1.3% | -8.8% | 29 | FAIL |
| ETHUSDT | 0.703 | +21.5% | -8.9% | 21 | PASS |
| SOLUSDT | 0.075 | +5.6% | -16.1% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume spike.
# Alligator lines: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift).
# Long when Lips > Teeth > Jaw (bullish alignment) and price > Lips, with 1d uptrend and volume > 1.5x 20-period average.
# Short when Lips < Teeth < Jaw (bearish alignment) and price < Lips, with 1d downtrend and volume > 1.5x average.
# Williams Alligator identifies trend presence and direction; 1d trend filter ensures higher timeframe alignment.
# Volume spike confirms institutional participation. Designed for ~20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Smoothed Moving Average (SMMA) with specific periods and shifts
    # Jaw: 13-period SMMA, 8 bars ahead
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)  # shift 8 bars into future
    
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)  # shift 5 bars into future
    
    # Lips: 5-period SMMA, 3 bars ahead
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)  # shift 3 bars into future
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Lips > Teeth > Jaw (bullish alignment), price > Lips, 1d uptrend, volume filter
        if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
            close[i] > lips[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Lips < Teeth < Jaw (bearish alignment), price < Lips, 1d downtrend, volume filter
        elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
              close[i] < lips[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 18:59
