#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 12h EMA50 Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels from daily pivot act as stronger support/resistance.
Breakouts above H3 or below L3 with 12h EMA trend alignment and volume spikes
capture strong moves with fewer false signals than R1/S1. Uses 4h timeframe with 12h HTF
for trend filter and volume confirmation. Targets 75-200 trades over 4 years.
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
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Get 12h data for EMA trend and volume MA (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 12h
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_20_12h[i] = np.mean(df_12h['volume'].values[i-19:i+1])
    
    # Align to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
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
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        vol_ma_12h = vol_ma_20_12h_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period 4h average AND
        # 12h volume > 2.0 * 20-period 12h average (strong volume on both timeframes)
        volume_confirm_4h = curr_volume > 2.0 * vol_ma_4h
        volume_confirm_12h = prices['volume'].values[i] > 2.0 * vol_ma_12h if hasattr(prices['volume'].values, '__getitem__') else False
        # Fallback: use 4h volume confirmation only if 12h volume data not directly accessible
        volume_confirm = volume_confirm_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3, above 12h EMA, volume confirmation
            long_entry = (curr_close > camarilla_h3 and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below L3, below 12h EMA, volume confirmation
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
            # Exit: price falls below L3 OR below 12h EMA
            if curr_close < camarilla_l3 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 OR above 12h EMA
            if curr_close > camarilla_h3 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0