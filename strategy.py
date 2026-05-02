#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# (-80 for long, -20 for short) with trend filter (price > EMA34 for long, < EMA34 for short)
# and volume spike (>1.5x average) to confirm institutional participation.
# This strategy avoids overtrading by requiring multiple confluence factors and uses
# discrete position sizing (0.25) to minimize fee drag. Tested on 6h timeframe with
# 1d HTF for trend and Williams %R calculation, suitable for both bull and bear markets
# due to its mean-reversion nature with trend alignment.

name = "6h_WilliamsR_Reversal_1dEMA34_Volume"
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
    
    # Calculate Williams %R and EMA34 from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    hl_range_1d = highest_high_1d - lowest_low_1d
    hl_range_1d = np.where(hl_range_1d == 0, 1e-10, hl_range_1d)
    williams_r_1d = ((highest_high_1d - close_1d) / hl_range_1d) * -100
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Williams %R and EMA34 to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.5x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold reversal) AND price > EMA34 AND volume spike
            if (williams_r_1d_aligned[i] > -80 and 
                williams_r_1d_aligned[i-1] <= -80 and  # Cross above -80
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND price < EMA34 AND volume spike
            elif (williams_r_1d_aligned[i] < -20 and 
                  williams_r_1d_aligned[i-1] >= -20 and  # Cross below -20
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -20 (overbought) OR price < EMA34
            if williams_r_1d_aligned[i] > -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -80 (oversold) OR price > EMA34
            if williams_r_1d_aligned[i] < -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals