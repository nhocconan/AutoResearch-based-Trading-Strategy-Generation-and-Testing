# Strategy: 6h_WilliamsAlligator_1dEMA34_VolumeConfirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.337 | +38.2% | -14.5% | 182 | PASS |
| ETHUSDT | 0.354 | +42.5% | -12.2% | 171 | PASS |
| SOLUSDT | 0.634 | +92.3% | -20.1% | 174 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.333 | -8.2% | -9.2% | 70 | FAIL |
| ETHUSDT | 0.001 | +5.0% | -14.0% | 62 | PASS |
| SOLUSDT | -0.373 | -1.8% | -15.0% | 58 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator (Jaws=13, Teeth=8, Lips=5 SMAs) identifies trend direction.
# In bull markets: Lips > Teeth > Jaws = bullish alignment.
# In bear markets: Lips < Teeth < Jaws = bearish alignment.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume > 1.5x average confirms trend strength.
# Works in both bull and bear markets by following the Alligator's alignment.
# Uses discrete position sizing (0.25) to minimize fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h data
    jaws_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaws = pd.Series(close).rolling(window=jaws_period, min_periods=jaws_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(jaws_period, teeth_period, lips_period), n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaws
            bullish = lips[i] > teeth[i] and teeth[i] > jaws[i]
            # Bearish alignment: Lips < Teeth < Jaws
            bearish = lips[i] < teeth[i] and teeth[i] < jaws[i]
            
            # Long: Bullish alignment + above 1d EMA + volume spike
            if bullish and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + below 1d EMA + volume spike
            elif bearish and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment changes or price crosses 1d EMA
            if position == 1:
                # Exit long: Bearish alignment or price below 1d EMA
                bearish = lips[i] < teeth[i] and teeth[i] < jaws[i]
                if bearish or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bullish alignment or price above 1d EMA
                bullish = lips[i] > teeth[i] and teeth[i] > jaws[i]
                if bullish or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 10:39
