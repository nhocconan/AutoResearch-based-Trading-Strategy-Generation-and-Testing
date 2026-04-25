#!/usr/bin/env python3
"""
1d Williams %R Mean Reversion + 1w EMA34 Trend + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions. In strong weekly trends (price > weekly EMA34), 
mean reversion from extreme %R levels (< -80 for long, > -20 for short) with volume confirmation (>1.5x 20-period volume MA) 
provides high-probability entries. Weekly trend filter ensures alignment with higher timeframe momentum, reducing 
counter-trend whipsaw. Designed for 1d timeframe targeting 30-100 total trades over 4 years. Works in both bull and 
bear markets via weekly trend filter and volume confirmation.
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 weeks for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Williams %R(14) for mean reversion signals (1d)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    williams_r = np.full(n, np.nan)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and Williams %R
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        williams_r_val = williams_r[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for mean reversion signals from extreme Williams %R levels
            # Long: Williams %R < -80 (oversold) with volume confirmation in uptrend
            long_signal = (williams_r_val < -80) and volume_confirm and uptrend
            # Short: Williams %R > -20 (overbought) with volume confirmation in downtrend
            short_signal = (williams_r_val > -20) and volume_confirm and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit conditions: Williams %R returns to neutral (> -50) OR trend turns down
            if williams_r_val > -50 or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit conditions: Williams %R returns to neutral (< -50) OR trend turns up
            if williams_r_val < -50 or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_%R_MeanReversion_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0