#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 4h timeframe, use Donchian(20) breakout for entry, with 1d EMA for trend filter and volume confirmation for institutional participation. Exit on opposite Donchian breakout or trend reversal. This strategy targets strong trending moves with volume confirmation, reducing false signals and trade frequency. Works in bull/bear via trend filter and breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA on 1d data
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align EMA to 4h timeframe
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels on 4h (20-period high/low)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_4h[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_1d_4h[i]
        downtrend = close[i] < ema_1d_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price breaks below lower Donchian channel
            if close[i] < low_min[i]:
                exit_long = True
            # Exit if trend turns down
            elif downtrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price breaks above upper Donchian channel
            if close[i] > high_max[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above upper Donchian channel with uptrend and volume confirmation
            if close[i] > high_max[i] and close[i-1] <= high_max[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below lower Donchian channel with downtrend and volume confirmation
            if close[i] < low_min[i] and close[i-1] >= low_min[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals