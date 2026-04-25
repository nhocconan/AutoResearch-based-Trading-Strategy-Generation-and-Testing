#!/usr/bin/env python3
"""
12h Camarilla R1/S1 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla R1/S1 levels act as strong daily support/resistance. 
Breakout above R1 with 1d uptrend (price > EMA34) and volume spike = bullish continuation.
Breakdown below S1 with 1d downtrend (price < EMA34) and volume spike = bearish continuation.
Uses 12h timeframe with 1d HTF for trend and Camarilla calculation. Targets 12-37 trades/year.
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
    
    # Get 1d data for Camarilla pivot and EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    camarilla_range = df_1d['high'].values - df_1d['low'].values
    camarilla_r1 = df_1d['close'].values + camarilla_range * 1.1 / 12
    camarilla_s1 = df_1d['close'].values - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h (previous day's levels available after 1d close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 10-period volume MA for 12h volume confirmation
    vol_ma_10_12h = np.full(n, np.nan)
    for i in range(10, n):
        vol_ma_10_12h[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_10_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_ma_12h = vol_ma_10_12h[i]
        
        # Volume confirmation: current 12h volume > 2.0 * 10-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price > EMA34 (uptrend) AND breaks above R1 AND volume confirmation
            long_entry = (curr_close > ema_trend and 
                         curr_high > r1_level and 
                         volume_confirm)
            # Short: price < EMA34 (downtrend) AND breaks below S1 AND volume confirmation
            short_entry = (curr_close < ema_trend and 
                          curr_low < s1_level and 
                          volume_confirm)
            
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
            # Exit: price falls below EMA34 OR breaks below S1 (failed breakout)
            if curr_close < ema_trend or curr_low < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above EMA34 OR breaks above R1 (failed breakdown)
            if curr_close > ema_trend or curr_high > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0