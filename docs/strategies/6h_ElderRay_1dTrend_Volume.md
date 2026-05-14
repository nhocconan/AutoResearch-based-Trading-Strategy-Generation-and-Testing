# Strategy: 6h_ElderRay_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.487 | +46.8% | -9.7% | 161 | PASS |
| ETHUSDT | 0.277 | +36.1% | -17.9% | 144 | PASS |
| SOLUSDT | 1.321 | +250.2% | -21.9% | 120 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.709 | -2.0% | -6.5% | 69 | FAIL |
| ETHUSDT | 0.128 | +7.3% | -8.6% | 54 | PASS |
| SOLUSDT | -0.163 | +1.9% | -15.7% | 52 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray measures bullish/bearish power by comparing high/low to EMA.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# We go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising,
# aligned with 1d EMA(34) trend and confirmed by volume spike.
# Designed for low trade frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_ElderRay_1dTrend_Volume"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA(13) for Elder Ray on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA
    bear_power = ema13 - low   # Bear Power: EMA - Low
    
    # Slope of Bull/Bear Power (1-period change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        bp = bull_power[i]
        bp_slope = bull_power_slope[i]
        bep = bear_power[i]
        bep_slope = bear_power_slope[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising + uptrend + volume spike
            if (bp > 0 and bp_slope > 0 and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0 and rising + downtrend + volume spike
            elif (bep > 0 and bep_slope > 0 and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR price breaks below trend
            if (bp <= 0 or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns negative OR price breaks above trend
            if (bep <= 0 or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 12:15
