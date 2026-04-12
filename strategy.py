#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla long/short bias from 1d pivot + volume confirmation
    # Works in bull/bear by fading extreme intraday moves at key pivot levels.
    # Uses 12h timeframe to reduce frequency, volume filter to avoid false signals.
    # Target: 15-25 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    camarilla_H5 = np.full(len(df_1d), np.nan)
    camarilla_H4 = np.full(len(df_1d), np.nan)
    camarilla_H3 = np.full(len(df_1d), np.nan)
    camarilla_L3 = np.full(len(df_1d), np.nan)
    camarilla_L4 = np.full(len(df_1d), np.nan)
    camarilla_L5 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's data for today's levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_H5[i] = prev_close + 1.1 * range_val * 1.1
        camarilla_H4[i] = prev_close + 1.1 * range_val * 0.5
        camarilla_H3[i] = prev_close + 1.1 * range_val * 0.25
        camarilla_L3[i] = prev_close - 1.1 * range_val * 0.25
        camarilla_L4[i] = prev_close - 1.1 * range_val * 0.5
        camarilla_L5[i] = prev_close - 1.1 * range_val * 1.1
    
    # Align Camarilla levels to 12h timeframe
    H5_12h = align_htf_to_ltf(prices, df_1d, camarilla_H5)
    H4_12h = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    L5_12h = align_htf_to_ltf(prices, df_1d, camarilla_L5)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion at Camarilla H3/L3 levels
        # Long when price touches L3 with volume, short when touches H3 with volume
        long_signal = (low[i] <= L3_12h[i] + 0.1 * (H3_12h[i] - L3_12h[i])) and volume_filter[i]
        short_signal = (high[i] >= H3_12h[i] - 0.1 * (H3_12h[i] - L3_12h[i])) and volume_filter[i]
        
        # Exit when price moves back toward center or opposite extreme
        long_exit = (close[i] >= (H3_12h[i] + L3_12h[i]) / 2) or (high[i] >= H4_12h[i])
        short_exit = (close[i] <= (H3_12h[i] + L3_12h[i]) / 2) or (low[i] <= L4_12h[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_mean_reversion_v1"
timeframe = "12h"
leverage = 1.0