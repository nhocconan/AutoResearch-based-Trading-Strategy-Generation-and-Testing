#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Williams Fractals identify potential swing points: bearish fractal (high surrounded by two lower highs),
# bullish fractal (low surrounded by two higher lows). Breakout above recent bearish fractal = long signal,
# breakdown below recent bullish fractal = short signal. 1d EMA34 provides trend filter, volume spike confirms.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsFractal_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Fractals on 12h data (requires 5 bars: n-2, n-1, n, n+1, n+2)
    # We'll compute them inside the loop to avoid look-ahead, using only past data
    # Bearish fractal: high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    # Since we can't see future bars, we'll use the fractal from 2 bars ago (confirmed by two subsequent bars)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track recent fractal levels
    recent_bearish_fractal = np.nan  # resistance level from bearish fractal
    recent_bullish_fractal = np.nan   # support level from bullish fractal
    
    start_idx = max(34, 20)  # 1d EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Update fractals using data from 2 bars ago (i-2) which is now confirmed
        if i >= 2:
            idx = i - 2
            # Check for bearish fractal at idx: high[idx] is highest of idx-2,idx-1,idx,idx+1,idx+2
            if (idx >= 2 and idx + 2 < n and 
                high[idx] > high[idx-2] and high[idx] > high[idx-1] and 
                high[idx] > high[idx+1] and high[idx] > high[idx+2]):
                recent_bearish_fractal = high[idx]
            # Check for bullish fractal at idx: low[idx] is lowest of idx-2,idx-1,idx,idx+1,idx+2
            if (idx >= 2 and idx + 2 < n and 
                low[idx] < low[idx-2] and low[idx] < low[idx-1] and 
                low[idx] < low[idx+1] and low[idx] < low[idx+2]):
                recent_bullish_fractal = low[idx]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: close below 1d EMA34 OR volume spike breakdown
            if curr_close < curr_ema_1d or (curr_close < recent_bullish_fractal and not np.isnan(recent_bullish_fractal)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above 1d EMA34 OR volume spike breakout
            if curr_close > curr_ema_1d or (curr_close > recent_bearish_fractal and not np.isnan(recent_bearish_fractal)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: close above bearish fractal AND above 1d EMA34 AND volume spike
            bullish_breakout = (not np.isnan(recent_bearish_fractal) and 
                               curr_close > recent_bearish_fractal and 
                               curr_close > curr_ema_1d and 
                               vol_spike)
            # Short entry: close below bullish fractal AND below 1d EMA34 AND volume spike
            bearish_breakdown = (not np.isnan(recent_bullish_fractal) and 
                                curr_close < recent_bullish_fractal and 
                                curr_close < curr_ema_1d and 
                                vol_spike)
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakdown:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals