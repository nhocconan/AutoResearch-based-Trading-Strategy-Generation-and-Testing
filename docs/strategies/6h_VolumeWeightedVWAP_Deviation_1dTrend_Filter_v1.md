# Strategy: 6h_VolumeWeightedVWAP_Deviation_1dTrend_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.132 | +15.5% | -10.5% | 90 | FAIL |
| ETHUSDT | 0.235 | +32.9% | -10.2% | 126 | PASS |
| SOLUSDT | -0.440 | -15.1% | -33.1% | 133 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.643 | +16.0% | -7.0% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_VolumeWeightedVWAP_Deviation_1dTrend_Filter_v1
Hypothesis: Trade 6h deviations from volume-weighted VWAP with 1d trend filter. In bullish 1d trend, long when price < VWAP (mean reversion); in bearish 1d trend, short when price > VWAP (mean reversion). Volume-weighted VWAP acts as dynamic fair value. Uses 6h bar's VWAP calculated from typical price and volume. Requires deviation > 1.5% from VWAP to avoid noise. Target: 50-150 total trades over 4 years = 12-37/year.
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
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h VWAP (volume-weighted average price) for each 6h bar
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    typical_price_6h = (df_6h['high'].values + df_6h['low'].values + df_6h['close'].values) / 3.0
    vwap_6h = (typical_price_6h * df_6h['volume'].values).cumsum() / df_6h['volume'].values.cumsum()
    # Handle first bar where cumulative volume might be zero
    vwap_6h = np.where(df_6h['volume'].values.cumsum() == 0, typical_price_6h, vwap_6h)
    
    # Align 6h VWAP to 6h timeframe
    vwap_6h_aligned = align_htf_to_ltf(prices, df_6h, vwap_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vwap_6h_aligned[i]) or
            vwap_6h_aligned[i] == 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate percentage deviation from VWAP
        deviation_pct = (close[i] - vwap_6h_aligned[i]) / vwap_6h_aligned[i] * 100.0
        
        # Determine 1d HTF trend using EMA34
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Mean reversion entries: fade deviation from VWAP in direction of 1d trend
            # In bullish 1d trend: long when price < VWAP (oversold)
            # In bearish 1d trend: short when price > VWAP (overbought)
            long_setup = (deviation_pct < -1.5) and htf_1d_bullish  # Oversold in bullish trend
            short_setup = (deviation_pct > 1.5) and htf_1d_bearish   # Overbought in bearish trend
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to VWAP or trend reverses
            exit_signal = (deviation_pct >= -0.5) or (not htf_1d_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to VWAP or trend reverses
            exit_signal = (deviation_pct <= 0.5) or htf_1d_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_VolumeWeightedVWAP_Deviation_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 17:13
