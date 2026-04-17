#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R1/S1 breakout + 1w ADX trend filter + volume confirmation.
Long when price breaks above Camarilla R1 level with 1w ADX > 25 (trending) and volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 level with 1w ADX > 25 and volume > 1.5x 20-period average.
Uses discrete position sizing (0.25) to minimize fee churn. Designed to capture trending moves with volume confirmation
while avoiding choppy markets via ADX filter. Works in both bull (breakouts above R1) and bear (breakdowns below S1).
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d typical price for Camarilla
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels (R1, S1)
    camarilla_r1 = typical_price_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = typical_price_1d - (range_1d * 1.1 / 12)
    
    # Calculate 1w ADX (14-period)
    def calculate_adx(high_vals, low_vals, close_vals, window):
        plus_dm = np.zeros_like(high_vals)
        minus_dm = np.zeros_like(low_vals)
        tr = np.zeros_like(high_vals)
        
        for i in range(1, len(high_vals)):
            plus_dm[i] = max(0, high_vals[i] - high_vals[i-1])
            minus_dm[i] = max(0, low_vals[i-1] - low_vals[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(
                high_vals[i] - low_vals[i],
                abs(high_vals[i] - close_vals[i-1]),
                abs(low_vals[i] - close_vals[i-1])
            )
        
        # Handle first bar
        tr[0] = high_vals[0] - low_vals[0]
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Handle division by zero
        adx = np.where((plus_di + minus_di) == 0, 0, adx)
        return adx
    
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(adx_14_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: 1w ADX > 25 indicates trending market
        trending = adx_14_1w_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume and trend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_confirmed and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume and trend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_confirmed and 
                  trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla S1 (opposite side)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Camarilla R1 (opposite side)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Breakout_1wADX_Volume_Confirm"
timeframe = "12h"
leverage = 1.0