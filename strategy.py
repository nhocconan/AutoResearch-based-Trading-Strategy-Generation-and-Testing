#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h timeframe with Camarilla R1/S1 breakout from previous 1d pivot levels, 
1d EMA34 trend filter, and volume confirmation (>1.8x 20-bar avg volume). 
Long when price breaks above R1 in 1d uptrend with volume spike; short when breaks below S1 in 1d downtrend with volume spike. 
Exits on opposite level break or trend reversal. 
Designed for BTC/ETH to work in bull/bear via structure (Camarilla pivots) with trend/volume filters. 
Target trades: 75-200 total over 4 years (19-50/year).
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
    
    # 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (R1, S1, PP)
    # Formula: PP = (H + L + C) / 3
    #          R1 = PP + (H - L) * 1.1 / 2
    #          S1 = PP - (H - L) * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 in 1d uptrend with volume spike
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
            long_signal = (curr_close > r1_aligned[i]) and \
                         (close_1d_aligned[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below S1 in 1d downtrend with volume spike
            short_signal = (curr_close < s1_aligned[i]) and \
                          (close_1d_aligned[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down
            if (curr_close < s1_aligned[i]) or \
               (close_1d_aligned[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up
            if (curr_close > r1_aligned[i]) or \
               (close_1d_aligned[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0