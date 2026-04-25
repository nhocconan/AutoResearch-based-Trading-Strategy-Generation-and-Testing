#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg). Enters long when price breaks above R1 in 1d uptrend, short when breaks below S1 in 1d downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (using prior 1d bar to avoid look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Camarilla R1, S1, R3, S3
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + camarilla_range * 1.0/12
    s1 = prev_close_1d - camarilla_range * 1.0/12
    r3 = prev_close_1d + camarilla_range * 3.0/12
    s3 = prev_close_1d - camarilla_range * 3.0/12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 1d EMA34 data
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 in 1d uptrend with volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and (close_1d[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: break below S1 in 1d downtrend with volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and (close_1d[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price moves below S3 (reversal) OR trend turns down
            if (close[i] < s3_aligned[i]) or (close_1d[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price moves above R3 (reversal) OR trend turns up
            if (close[i] > r3_aligned[i]) or (close_1d[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0