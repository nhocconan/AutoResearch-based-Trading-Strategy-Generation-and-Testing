# Strategy: 12h_Camarilla_R1S1_1dEMA34_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.079 | +23.6% | -6.2% | 79 | PASS |
| ETHUSDT | 0.011 | +20.8% | -8.0% | 67 | PASS |
| SOLUSDT | 0.159 | +28.4% | -20.6% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.438 | -4.4% | -8.0% | 34 | FAIL |
| ETHUSDT | 0.561 | +12.7% | -7.2% | 26 | PASS |
| SOLUSDT | -1.025 | -5.6% | -14.7% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long: price breaks above Camarilla R1 (1d) + price > 1d EMA34 + volume > 2.0x 20-period avg volume
- Short: price breaks below Camarilla S1 (1d) + price < 1d EMA34 + volume > 2.0x 20-period avg volume
- Exit: ATR trailing stop (2.0x ATR from extreme) OR Camarilla breakout in opposite direction
- Uses 1d EMA34 as trend filter for better regime adaptation on 12h timeframe
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 2.0x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels (R1, S1) on 12h data using previous day's OHLC
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # But we need to use previous day's OHLC for today's levels
    # Since we're on 12h timeframe, we need to get daily OHLC from 1d data
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from 1d data (using previous day's OHLC)
    # For each 1d bar, calculate levels for the next day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 for each day (based on previous day's OHLC)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Shift to get previous day's levels for current day's trading
    camarilla_r1_prev = np.concatenate([[np.nan], camarilla_r1[:-1]])
    camarilla_s1_prev = np.concatenate([[np.nan], camarilla_s1[:-1]])
    
    # Load 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_prev)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_prev)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20, 34)  # Need 20 for volume MA, 14 for ATR, 20 for Camarilla (implicit), 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous day's levels)
        breakout_up = close[i] > camarilla_r1_aligned[i]  # Break above Camarilla R1
        breakout_down = close[i] < camarilla_s1_aligned[i]  # Break below Camarilla S1
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + price > 1d EMA34 + volume spike
            if breakout_up and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Camarilla breakout down + price < 1d EMA34 + volume spike
            elif breakout_down and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Camarilla breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            breakout_down_exit = close[i] < camarilla_s1_aligned[i]
            
            if trailing_stop_long or breakout_down_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from short extreme (trailing stop)
            # 2. Camarilla breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            breakout_up_exit = close[i] > camarilla_r1_aligned[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 19:49
