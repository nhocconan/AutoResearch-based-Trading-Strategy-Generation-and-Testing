# Strategy: 6h_ElderRay_BullPower_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.107 | +24.7% | -16.3% | 116 | PASS |
| ETHUSDT | 0.302 | +41.9% | -14.5% | 117 | PASS |
| SOLUSDT | 1.027 | +222.4% | -30.4% | 117 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.603 | -2.6% | -7.7% | 46 | FAIL |
| ETHUSDT | 0.551 | +17.2% | -9.4% | 40 | PASS |
| SOLUSDT | -0.018 | +3.4% | -14.1% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullPower_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Elder Ray components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for power calculation (standard period)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe (using previous day's values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # EMA26 for trend filter (longer term)
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema26_1d)
    
    # Volume filter: current volume > 1.5 * 6-period average (1.5 days on 6h chart)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_filter = volume > (1.5 * vol_ma_6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema26_1d_aligned[i]) or np.isnan(vol_ma_6[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema_val = ema26_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price above EMA26 AND volume filter
            if bull_val > 0 and close_val > ema_val and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price below EMA26 AND volume filter
            elif bear_val < 0 and close_val < ema_val and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR price crosses below EMA26
            if bull_val <= 0 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive OR price crosses above EMA26
            if bear_val >= 0 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-18 22:43
