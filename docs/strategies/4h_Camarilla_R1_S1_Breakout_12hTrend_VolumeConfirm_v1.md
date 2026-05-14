# Strategy: 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.031 | +19.2% | -11.8% | 319 | FAIL |
| ETHUSDT | 0.079 | +23.5% | -13.3% | 296 | PASS |
| SOLUSDT | 0.647 | +77.0% | -20.7% | 269 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.869 | +18.0% | -5.5% | 109 | PASS |
| SOLUSDT | 0.807 | +17.2% | -9.1% | 95 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume confirmation. 
Camarilla R1/S1 provide intraday support/resistance levels. In bull markets: buy when price breaks above R1 and price > 12h EMA50. 
In bear markets: sell when price breaks below S1 and price < 12h EMA50. 
Requires volume > 1.8x 20-period average for confirmation to avoid false breakouts. 
Exit on opposite Camarilla level touch (R1 for shorts, S1 for longs) or trend reversal. 
Position size: 0.25 to balance reward and risk. 
Target: 80-180 total trades over 4 years = 20-45/year. 
Intraday levels with HTF trend filter work in both bull and bear markets by aligning with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    # Daily Camarilla R1 and S1 (key intraday resistance/support)
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align daily Camarilla levels to 4h prices
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above daily Camarilla R1 + 12h uptrend + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_12h_bullish and volume_confirm
            
            # Short setup: price breaks below daily Camarilla S1 + 12h downtrend + volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and htf_12h_bearish and volume_confirm
            
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
            # Exit: price touches daily Camarilla S1 (stop) OR 12h trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches daily Camarilla R1 (stop) OR 12h trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 16:40
