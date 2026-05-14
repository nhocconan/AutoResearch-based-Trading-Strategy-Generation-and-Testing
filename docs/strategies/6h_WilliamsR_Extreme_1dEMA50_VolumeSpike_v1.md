# Strategy: 6h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.373 | +36.4% | -7.9% | 72 | KEEP |
| ETHUSDT | 0.298 | +34.0% | -10.8% | 49 | KEEP |
| SOLUSDT | 0.796 | +92.3% | -12.5% | 39 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.343 | +3.0% | -4.6% | 21 | DISCARD |
| ETHUSDT | 0.557 | +13.4% | -6.6% | 21 | KEEP |
| SOLUSDT | -0.754 | -4.3% | -11.8% | 18 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend Filter with Volume Spike.
- Primary timeframe: 6h for execution, HTF: 1d for EMA50 trend filter.
- Entry: Williams %R(14) crosses above -20 from below (long) or below -80 from above (short) on 6h close, with volume > 2.0x 20-period volume MA.
- Direction filter: only long when 6h close > 1d EMA50, only short when 6h close < 1d EMA50.
- Williams %R extremes identify overbought/oversold conditions; EMA50 provides trend filter to avoid counter-trend trades.
- Volume confirmation reduces false signals.
- Exit: opposite Williams %R extreme touch (long exits at -80, short exits at -20) or trend filter reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold dips in uptrend, in bear via selling overbought rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    if len(high) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r[highest_high == lowest_low] = -50
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need 1d EMA50, Williams %R(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below (oversold bounce) with volume spike AND uptrend (close > 1d EMA50)
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and 
                close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above (overbought rejection) with volume spike AND downtrend (close < 1d EMA50)
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns below -80 (mean reversion) or trend reversal
            if williams_r[i] < -80 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns above -20 (mean reversion) or trend reversal
            if williams_r[i] > -20 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-30 21:51
