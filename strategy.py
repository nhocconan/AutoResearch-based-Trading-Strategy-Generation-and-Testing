#!/usr/bin/env python3
"""
1d Camarilla H3L3 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) from weekly data act as strong support/resistance. 
Breakout above H3 or below L3 with volume confirmation and weekly EMA50 trend filter captures 
multi-day momentum moves. Works in bull (long on H3 break) and bear (short on L3 break). 
Volume spike ensures institutional participation. Target: 10-25 trades/year on 1d.
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
    
    # Get 1w data for EMA50 trend and Camarilla pivots (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels on 1w (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    camarilla_h3 = close_1w_arr + 1.1 * (high_1w - low_1w) / 2
    camarilla_l3 = close_1w_arr - 1.1 * (high_1w - low_1w) / 2
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Calculate 10-period volume MA for volume confirmation
    vol_ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        vol_ma_10[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, volume MA
    start_idx = max(50, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_10[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_ma = vol_ma_10[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 10-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > H3, above weekly EMA50, volume confirmation
            long_entry = (curr_close > h3_level) and (curr_close > ema_50_val) and volume_confirm
            # Short: price < L3, below weekly EMA50, volume confirmation
            short_entry = (curr_close < l3_level) and (curr_close < ema_50_val) and volume_confirm
            
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
            # Exit: price crosses below weekly EMA50 OR price breaks below L3 (stop and reverse)
            if curr_close < ema_50_val or curr_close < l3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly EMA50 OR price breaks above H3 (stop and reverse)
            if curr_close > ema_50_val or curr_close > h3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0