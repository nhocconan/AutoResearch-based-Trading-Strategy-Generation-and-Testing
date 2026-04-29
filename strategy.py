#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from below, 1d EMA34 up-trend, volume > 1.8x average
# Short when Williams %R crosses below -20 from above, 1d EMA34 down-trend, volume > 1.8x average
# Exit when Williams %R crosses -50 (mean reversion to midpoint)
# Uses discrete position sizing (0.25) and moderate volume filter to target 12-37 trades/year.
# Williams %R identifies overbought/oversold conditions with mean reversion tendency.
# 1d EMA34 ensures we trade with the higher timeframe trend in both bull and bear markets.
# Volume confirmation filters out low-momentum breakouts.

name = "12h_WilliamsR_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Williams %R calculation (period=14)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_12h['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 12h Williams %R to 12h timeframe (no additional delay needed)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Warmup for volume, 1d EMA34, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Previous bar values for crossover detection
        prev_williams_r = williams_r_aligned[i-1]
        prev_close = close[i-1]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (mean reversion)
            if prev_williams_r > -50 and curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (mean reversion)
            if prev_williams_r < -50 and curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirmed = volume[i] > 1.8 * curr_vol_ma
            
            # Long when Williams %R crosses above -80 from below, 1d EMA34 up-trend, volume confirmed
            if (prev_williams_r <= -80 and curr_williams_r > -80 and 
                curr_close > curr_ema34_1d and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 from above, 1d EMA34 down-trend, volume confirmed
            elif (prev_williams_r >= -20 and curr_williams_r < -20 and 
                  curr_close < curr_ema34_1d and vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals