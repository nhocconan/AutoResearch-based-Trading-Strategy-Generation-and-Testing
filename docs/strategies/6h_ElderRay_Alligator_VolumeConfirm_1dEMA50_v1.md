# Strategy: 6h_ElderRay_Alligator_VolumeConfirm_1dEMA50_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.280 | +32.1% | -9.3% | 207 | PASS |
| ETHUSDT | 0.060 | +22.5% | -13.7% | 195 | PASS |
| SOLUSDT | 0.437 | +55.9% | -24.7% | 185 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.311 | +3.3% | -5.2% | 69 | FAIL |
| ETHUSDT | 0.064 | +6.4% | -6.7% | 72 | PASS |
| SOLUSDT | -0.383 | -0.2% | -9.9% | 59 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + Williams Alligator with volume confirmation and 1d trend filter.
- Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
- Williams Alligator (Jaw/Teeth/Lips SMAs) identifies trend vs ranging markets
- Long when Bull Power > 0, Lips > Teeth > Jaw (bullish alignment), and volume > 1.5x average
- Short when Bear Power > 0, Lips < Teeth < Jaw (bearish alignment), and volume > 1.5x average
- 1d EMA50 as higher timeframe trend filter (avoid counter-trend trades)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
- Works in both bull/bear via 1d trend filter and volatility-adjusted signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA13 for Bull/Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # SMA13
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # SMA8
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # SMA5
    
    # Volume confirmation: > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30, 50)  # Elder Ray, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator alignment
        alligator_bull = lips[i] > teeth[i] and teeth[i] > jaw[i]  # Lips > Teeth > Jaw
        alligator_bear = lips[i] < teeth[i] and teeth[i] < jaw[i]  # Lips < Teeth < Jaw
        
        if position == 0:
            # Long: Bull Power > 0 AND alligator bullish AND price above 1d EMA50 AND volume confirmation
            if bull_power[i] > 0 and alligator_bull and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND alligator bearish AND price below 1d EMA50 AND volume confirmation
            elif bear_power[i] > 0 and alligator_bear and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR alligator alignment breaks OR price crosses below 1d EMA50
            if bull_power[i] <= 0 or not alligator_bull or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 OR alligator alignment breaks OR price crosses above 1d EMA50
            if bear_power[i] <= 0 or not alligator_bear or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Alligator_VolumeConfirm_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 23:35
