#!/usr/bin/env python3
"""
12h Williams %R Extreme Reversal + Volume Spike + 1d EMA34 Trend Filter
Strategy: Enter long when Williams %R crosses above -20 from below with volume spike
          and price > 1d EMA34 (bullish trend). Enter short when Williams %R crosses
          below -80 from above with volume spike and price < 1d EMA34 (bearish trend).
          Exit when Williams %R returns to -50 (neutral) or trend weakens.
          Williams %R identifies overbought/oversold conditions; extreme readings
          often precede reversals. Works in both bull (buy oversold) and bear (sell overbought).
          Designed for low trade frequency with clear reversal edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R calculation and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on daily data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    rr = high_14 - low_14
    williams_r = np.where(rr != 0, ((high_14 - close_1d) / rr) * -100, -50)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for Williams %R calculation
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        williams_r_val = williams_r_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Williams %R cross signals
        williams_r_prev = williams_r_aligned[i-1] if i > 0 else williams_r_val
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below with volume spike and above daily EMA34
            if (williams_r_prev <= -20 and williams_r_val > -20 and 
                volume_spike[i] and price > ema_34):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above with volume spike and below daily EMA34
            elif (williams_r_prev >= -80 and williams_r_val < -80 and 
                  volume_spike[i] and price < ema_34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: Williams %R returns to -50 (neutral) or below EMA34 (trend change)
            if williams_r_val >= -50 or price < ema_34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: Williams %R returns to -50 (neutral) or above EMA34 (trend change)
            if williams_r_val <= -50 or price > ema_34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WilliamsR_Extreme_Reversal_VolumeSpike_1dEMA34"
timeframe = "12h"
leverage = 1.0