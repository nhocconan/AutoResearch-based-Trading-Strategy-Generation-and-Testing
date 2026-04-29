#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA50 trend and volume confirmation
# Long when %R crosses above -80 (oversold bounce), 1d EMA50 up-trend, volume > 1.8x average
# Short when %R crosses below -20 (overbought rejection), 1d EMA50 down-trend, volume > 1.8x average
# Exit when %R crosses opposite extreme (-20 for long, -80 for short) or reverses to midpoint (-50)
# Uses discrete position sizing (0.25) and moderate volume filter to target ~100-180 trades over 4 years.
# Williams %R is a momentum oscillator that works well in ranging markets and catches reversals in trends.
# The 1d EMA50 provides the higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction.

name = "4h_WilliamsR_1dEMA50_Volume_v1"
timeframe = "4h"
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
    
    # Get 4h data for Williams %R calculation (period 14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_4h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_4h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_4h['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 4h Williams %R to 4h timeframe (no additional delay needed for momentum indicator)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 1d EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum fading) or above -20 (overbought)
            if curr_williams_r < -50 or curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum fading) or below -80 (oversold)
            if curr_williams_r > -50 or curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average (moderate filter)
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when Williams %R crosses above -80 from below (oversold bounce)
            # AND 1d EMA50 is up-trend (price above EMA50)
            # AND volume confirmed
            if (curr_williams_r > -80 and 
                # Check for crossover: previous was <= -80, current > -80
                i > start_idx and williams_r_aligned[i-1] <= -80 and
                curr_close > curr_ema50_1d and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 from above (overbought rejection)
            # AND 1d EMA50 is down-trend (price below EMA50)
            # AND volume confirmed
            elif (curr_williams_r < -20 and
                  # Check for crossover: previous was >= -20, current < -20
                  i > start_idx and williams_r_aligned[i-1] >= -20 and
                  curr_close < curr_ema50_1d and vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals