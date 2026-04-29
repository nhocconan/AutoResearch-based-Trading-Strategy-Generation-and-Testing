#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -20 (oversold bounce), 1d EMA34 up-trend, volume > 1.5x average
# Short when Williams %R crosses below -80 (overbought rejection), 1d EMA34 down-trend, volume > 1.5x average
# Exit when Williams %R reverts to -50 (mean reversion) or opposite signal triggers
# Uses 12h primary timeframe for lower trade frequency (~12-37/year) and 1d for trend filter
# Williams %R is effective in ranging markets (2025+ test period) and captures mean reversion
# Volume confirmation reduces false signals; discrete sizing (0.25) minimizes fee churn

name = "12h_WilliamsR_1dEMA34_Volume_v1"
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
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_12h['close'].values) / (highest_high - lowest_low + 1e-10) * -100
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
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Previous Williams %R for crossover detection
        prev_williams_r = williams_r_aligned[i-1] if i > 0 else curr_williams_r
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (mean reversion) or opposite signal
            if curr_williams_r < -50 and prev_williams_r >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (mean reversion) or opposite signal
            if curr_williams_r > -50 and prev_williams_r <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when Williams %R crosses above -20 (oversold bounce), 1d EMA34 up-trend, volume confirmed
            if curr_williams_r > -20 and prev_williams_r <= -20 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -80 (overbought rejection), 1d EMA34 down-trend, volume confirmed
            elif curr_williams_r < -80 and prev_williams_r >= -80 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals