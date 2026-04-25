#!/usr/bin/env python3
"""
12h Camarilla R1S1 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: On 12h timeframe, Camarilla R1/S1 levels from weekly data act as strong support/resistance. 
Breakout above R1 or below S1 with volume confirmation and weekly EMA50 trend filter captures momentum moves.
Uses wider stops and fewer trades suitable for 12h timeframe (target 12-37 trades/year). Works in bull (long on R1 break) 
and bear (short on S1 break). Weekly EMA50 ensures we trade with the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend and Camarilla pivots (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels on weekly (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    camarilla_r1 = close_1w_arr + 1.1 * (high_1w - low_1w) / 12
    camarilla_s1 = close_1w_arr - 1.1 * (high_1w - low_1w) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Calculate 20-period volume MA for volume confirmation (on 12h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > R1, above weekly EMA50, volume confirmation
            long_entry = (curr_close > r1_level) and (curr_close > ema_50_val) and volume_confirm
            # Short: price < S1, below weekly EMA50, volume confirmation
            short_entry = (curr_close < s1_level) and (curr_close < ema_50_val) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below weekly EMA50 OR price breaks below S1 (stop and reverse)
            if curr_close < ema_50_val or curr_close < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly EMA50 OR price breaks above R1 (stop and reverse)
            if curr_close > ema_50_val or curr_close > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0