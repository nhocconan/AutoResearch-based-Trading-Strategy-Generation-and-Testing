#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels from daily pivot act as strong support/resistance.
Breakouts above H3 or below L3 with 1d EMA trend alignment and volume spikes
capture strong moves. Uses 4h timeframe with 1d HTF for trend filter and volume confirmation.
Targets 75-200 trades over 4 years (19-50/year) to avoid fee drag.
Works in both bull and bear markets by following the 1d EMA trend direction.
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
    
    # Get 1d data for Camarilla pivots and EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Calculate 34-period EMA on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d for volume confirmation
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    
    # Align to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 20-period volume MA for 4h volume spike
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period 4h average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3, above 1d EMA, volume confirmation
            long_entry = (curr_close > camarilla_h3 and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below L3, below 1d EMA, volume confirmation
            short_entry = (curr_close < camarilla_l3 and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below L3 OR below 1d EMA
            if curr_close < camarilla_l3 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 OR above 1d EMA
            if curr_close > camarilla_h3 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0