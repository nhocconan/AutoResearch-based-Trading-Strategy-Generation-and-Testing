#!/usr/bin/env python3
"""
4h Camarilla Pivot H3L3 Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts above H3 or below L3 with volume confirmation and aligned 1d EMA34 trend capture strong moves. Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakouts in downtrend) by using 1d EMA34 for trend filter. Volume spike (>2.0x 20-bar vol MA) confirms momentum. Designed for low trade frequency (~25-35/year) to avoid fee drag.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need 34 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    camarilla_h3 = close_1d_arr + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d_arr - 1.125 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, Camarilla, and volume MA
    start_idx = max(35, 20)  # 35 for EMA34 (34 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = curr_close > ema_34_val
        price_below_ema = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above H3 with uptrend + volume confirmation
            long_signal = (curr_close > camarilla_h3_val) and price_above_ema and volume_confirm
            # Short: break below L3 with downtrend + volume confirmation
            short_signal = (curr_close < camarilla_l3_val) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-tests H3 level OR trend reverses
            if (curr_close < camarilla_h3_val) or (curr_close < ema_34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-tests L3 level OR trend reverses
            if (curr_close > camarilla_l3_val) or (curr_close > ema_34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0