#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA200 trend + Volume Spike
# Williams %R identifies overbought/oversold conditions; extremes (<-95 or >-5) signal potential reversals
# 1d EMA200 provides strong trend filter to avoid counter-trend trades in bear/bull markets
# Volume > 2.5x 20-period average confirms institutional participation and reduces false signals
# Discrete position sizing (0.25) with exits on %R returning to normal levels (-50)
# Designed for ~12-30 trades/year on 6h to minimize fee drag while capturing strong reversals
# Works in bull/bear via trend filter - only trades in direction of 1d EMA200

name = "6h_WilliamsR_Extreme_1dEMA200_VolumeSpike_v1"
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams %R on 6h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, (highest_high - close) / hh_ll * -100, -50)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 200)  # Volume, Williams %R, and 1d EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        curr_ema200_1d = ema_200_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R returns to neutral territory (> -50)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral territory (< -50)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.5x 20-period average
            vol_confirmed = curr_volume > 2.5 * curr_vol_ma
            
            # Long when Williams %R extremely oversold (<-95), 1d EMA200 up-trend, volume confirmed
            if curr_williams_r < -95 and curr_close > curr_ema200_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R extremely overbought (> -5), 1d EMA200 down-trend, volume confirmed
            elif curr_williams_r > -5 and curr_close < curr_ema200_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals