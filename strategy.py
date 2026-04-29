#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold recovery), 12h EMA50 up-trend, volume > 1.5x average
# Short when Williams %R crosses below -20 (overbought rejection), 12h EMA50 down-trend, volume > 1.5x average
# Exit when Williams %R crosses opposite threshold (-20 for long exit, -80 for short exit)
# Uses discrete position sizing (0.25) and moderate volume filter to target ~75-150 trades over 4 years.
# Uses 12h for signal direction/trend, 6h only for Williams %R calculation and entry timing.

name = "6h_WilliamsR_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R on 6h data
    highest_high_14 = pd.Series(df_6h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_6h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - df_6h['close'].values) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 12h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Williams %R thresholds
        oversold = -80
        overbought = -20
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses above overbought threshold (-20)
            if curr_williams_r > overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below oversold threshold (-80)
            if curr_williams_r < oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when Williams %R crosses above oversold (-80) from below, 12h EMA50 up-trend, volume confirmed
            if (curr_williams_r > oversold and 
                williams_r_aligned[i-1] <= oversold and  # crossed above
                curr_close > curr_ema50_12h and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below overbought (-20) from above, 12h EMA50 down-trend, volume confirmed
            elif (curr_williams_r < overbought and 
                  williams_r_aligned[i-1] >= overbought and  # crossed below
                  curr_close < curr_ema50_12h and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals