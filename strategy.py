#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla pivot levels from 1d provide strong intraday support/resistance. 
Breakouts of R3/S3 levels with 1d EMA34 trend filter and volume spike confirmation 
capture institutional moves. Works in bull/bear by only taking breakouts in trend direction.
Target: 20-30 trades/year.
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
    
    # Get 1d data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d_prev) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    r3 = typical_price + 1.1 * range_1d / 2
    r1 = typical_price + 1.1 * range_1d / 6
    pp = typical_price
    s1 = typical_price - 1.1 * range_1d / 6
    s3 = typical_price - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume spike (>1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions (use previous bar's levels to avoid look-ahead)
        breakout_r3 = high[i] > r3_aligned[i-1]  # Break above R3
        breakout_s3 = low[i] < s3_aligned[i-1]   # Break below S3
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic: Only take breakouts in direction of 1d trend
        long_entry = breakout_r3 and trend_up and vol_confirm
        short_entry = breakout_s3 and trend_down and vol_confirm
        
        # Exit logic: Opposite Camarilla breakout or trend reversal
        long_exit = low[i] < s1_aligned[i-1] or not trend_up  # Break below S1 or trend down
        short_exit = high[i] > r1_aligned[i-1] or not trend_down  # Break above R1 or trend up
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0