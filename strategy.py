#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume spike confirmation.
# Williams %R(14) identifies overbought/oversold conditions. Long when %R crosses above -80 from below (oversold bounce),
# short when %R crosses below -20 from above (overbought reversal). Uses 1d EMA34 for trend alignment to avoid counter-trend trades,
# volume > 1.8x 20-bar average for confirmation. Discrete position sizing at ±0.25 to limit fee drag.
# Target: 80-160 total trades over 4 years (20-40/year). Works in both bull and bear markets by fading extremes with trend filter.

name = "6h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below, price above 1d EMA34, volume confirmation
            if (curr_williams_r > -80 and 
                curr_williams_r < -20 and  # Ensure we're not already in overbought
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                # Check for crossover: previous %R <= -80
                if i > start_idx and williams_r[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R crosses below -20 from above, price below 1d EMA34, volume confirmation
            elif (curr_williams_r < -20 and 
                  curr_williams_r > -80 and  # Ensure we're not already in oversold
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                # Check for crossover: previous %R >= -20
                if i > start_idx and williams_r[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -20 (overbought) or price crosses below 1d EMA34
            if curr_williams_r >= -20 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -80 (oversold) or price crosses above 1d EMA34
            if curr_williams_r <= -80 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals