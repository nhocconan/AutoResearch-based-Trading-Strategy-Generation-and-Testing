#102436
#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses Donchian 20-period breakouts on 12h chart with 1-day EMA34 trend filter and volume spikes (>2x average) to capture strong directional moves. Works in bull/bear by following 1-day trend direction. Targets 12-37 trades/year via strict breakout conditions and volume confirmation.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2x 24-period MA (4 days of 12h bars = 24 periods)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_24[i])
        
        # Breakout conditions at Donchian channels
        long_breakout = close[i] > high_max_20[i] and vol_confirm and uptrend
        short_breakout = close[i] < low_min_20[i] and vol_confirm and downtrend
        
        # Exit conditions: return to opposite Donchian level
        long_exit = close[i] < low_min_20[i]
        short_exit = close[i] > high_max_20[i]
        
        if long_breakout and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0