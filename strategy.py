#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w volume regime filter (low volume = range, high volume = trend)
    # Long: price breaks above upper band AND 1w volume > 1.5x 20-week average (trend confirmation)
    # Short: price breaks below lower band AND 1w volume > 1.5x 20-week average (trend confirmation)
    # Exit: price touches opposite Donchian band
    # Using 1d timeframe for optimal trade frequency (target 7-25/year), Donchian for structure,
    # 1w volume to filter ranging markets (low volume = chop, high volume = trend), and discrete position sizing (0.25)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly volume MA(20) for trend confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_20w = np.full(len(vol_1w), np.nan)
    for i in range(20, len(vol_1w)):
        vol_ma_20w[i] = np.mean(vol_1w[i-20:i])
    
    # Align weekly volume MA to 1d
    vol_ma_20w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20w)
    
    # Calculate daily Donchian channels (20-period)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_band[i] = np.max(high[i-20:i])
        lower_band[i] = np.min(low[i-20:i])
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20w_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime filter: weekly volume > 1.5x 20-week average indicates trending market
        trending_market = volume[i] > (1.5 * vol_ma_20w_aligned[i]) if not np.isnan(vol_ma_20w_aligned[i]) else False
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_band[i]
        breakout_lower = close[i] < lower_band[i]
        
        # Exit conditions: touch opposite band
        touch_lower = close[i] < lower_band[i]  # Exit long on lower band touch
        touch_upper = close[i] > upper_band[i]  # Exit short on upper band touch
        
        # Entry logic: Donchian breakout + trending market (volume confirmation)
        long_entry = breakout_upper and trending_market and volume_spike[i]
        short_entry = breakout_lower and trending_market and volume_spike[i]
        
        # Exit logic: opposite band touch
        long_exit = touch_lower
        short_exit = touch_upper
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0