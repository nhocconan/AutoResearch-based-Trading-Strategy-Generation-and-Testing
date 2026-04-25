#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeConfirm
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (price >/<- EMA34) and volume confirmation (>1.8x 20-bar avg). Enters long on break above R1 in 4h uptrend, short on break below S1 in 4h downtrend. Uses discrete sizing (0.20) to limit fee churn. Designed for 1h timeframe with ~20-50 trades/year, works in bull/bear by following 4h trend filter and session filter (08-20 UTC).
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
    
    # 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Previous day's high, low, close for Camarilla calculation
    # Use 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + (1.1 * camarilla_range / 12)
    s1 = close_1d - (1.1 * camarilla_range / 12)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need EMA34 and volume MA to be valid
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 in 4h uptrend with volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and (close_4h[i] > ema_34_4h_aligned[i]) and volume_spike[i]
            # Short: break below S1 in 4h downtrend with volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and (close_4h[i] < ema_34_4h_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: break below S1 OR trend turns down
            if (close[i] < s1_aligned[i]) or (close_4h[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: break above R1 OR trend turns up
            if (close[i] > r1_aligned[i]) or (close_4h[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0