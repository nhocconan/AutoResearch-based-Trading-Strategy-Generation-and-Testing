#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Donchian20_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    def calculate_donchian(high_arr, low_arr, window):
        n = len(high_arr)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        
        for i in range(window-1, n):
            upper[i] = np.max(high_arr[i-window+1:i+1])
            lower[i] = np.min(low_arr[i-window+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high_1w, low_1w, 20)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume spike (volume > 2.0 * 20-period average)
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma_1d * 2.0)
    
    # Align weekly indicators to 6h timeframe
    donch_upper_6h = align_htf_to_ltf(prices, df_1w, donch_upper)
    donch_lower_6h = align_htf_to_ltf(prices, df_1w, donch_lower)
    ema_34_1w_6h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    volume_spike_6h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_6h[i]) or np.isnan(donch_lower_6h[i]) or 
            np.isnan(ema_34_1w_6h[i]) or np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike_6h[i] > 0.5
        
        if position == 0:
            # Long when price breaks above weekly Donchian upper with volume AND weekly uptrend
            if close[i] > donch_upper_6h[i] and vol_confirm and close[i] > ema_34_1w_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below weekly Donchian lower with volume AND weekly downtrend
            elif close[i] < donch_lower_6h[i] and vol_confirm and close[i] < ema_34_1w_6h[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below weekly Donchian lower or trend turns down
            if close[i] < donch_lower_6h[i] or close[i] < ema_34_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above weekly Donchian upper or trend turns up
            if close[i] > donch_upper_6h[i] or close[i] > ema_34_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals