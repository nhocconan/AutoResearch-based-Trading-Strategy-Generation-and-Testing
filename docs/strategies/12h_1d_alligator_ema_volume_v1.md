# Strategy: 12h_1d_alligator_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.486 | -8.4% | -22.7% | 146 | DISCARD |
| ETHUSDT | 0.001 | +15.9% | -15.0% | 137 | KEEP |
| SOLUSDT | 0.820 | +150.3% | -26.0% | 126 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.132 | +7.3% | -13.1% | 43 | KEEP |
| SOLUSDT | -0.342 | -4.3% | -19.3% | 47 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA filter + volume confirmation
# - Primary signal: Williams Alligator (13,8,5 SMAs) - price above Alligator teeth (green) for long, below for short
# - Trend filter: 1d EMA50 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Alligator identifies trends, EMA50 filter ensures alignment with higher timeframe trend

name = "12h_1d_alligator_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    median_12h = df_12h['close'].rolling(window=30, min_periods=30).median().values
    jaw_12h = pd.Series(median_12h).ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw (13)
    teeth_12h = pd.Series(median_12h).ewm(span=8, adjust=False, min_periods=8).mean().values   # Teeth (8)
    lips_12h = pd.Series(median_12h).ewm(span=5, adjust=False, min_periods=5).mean().values    # Lips (5)
    
    # Align Alligator components to primary timeframe (completed 12h bar only)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Alligator teeth OR price crosses below 1d EMA50
            if close[i] < teeth_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Alligator teeth OR price crosses above 1d EMA50
            if close[i] > teeth_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with volume confirmation and 1d EMA50 filter
            # Long: price above Alligator lips AND teeth AND jaw (bullish alignment) AND volume regime AND price above 1d EMA50
            if (close[i] > lips_aligned[i] and 
                close[i] > teeth_aligned[i] and 
                close[i] > jaw_aligned[i] and 
                volume_regime[i] and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below Alligator lips AND teeth AND jaw (bearish alignment) AND volume regime AND price below 1d EMA50
            elif (close[i] < lips_aligned[i] and 
                  close[i] < teeth_aligned[i] and 
                  close[i] < jaw_aligned[i] and 
                  volume_regime[i] and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-10 00:04
