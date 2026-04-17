#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_Volume_Trend
Strategy: 1-hour breakout of daily Camarilla R1/S1 levels with volume confirmation and 4h trend filter.
Long: Price breaks above 1-day R1 + volume > 1.8x 20-period avg + price above 4h EMA34
Short: Price breaks below 1-day S1 + volume > 1.8x 20-period avg + price below 4h EMA34
Exit: Price returns to 1-hour VWAP
Position size: 0.20
Designed to capture breakouts aligned with 4h trend in both bull and bear markets.
Timeframe: 1h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h VWAP for exit
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate 1d OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_series_4h = pd.Series(close_4h)
    ema34_4h = close_series_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Camarilla levels and 4h EMA34 to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume confirmation (20-period MA on 1h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below 4h EMA34
        price_above_ema = close[i] > ema34_4h_aligned[i]
        price_below_ema = close[i] < ema34_4h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r1_aligned[i-1]  # break above previous day R1
        breakout_down = close[i] < camarilla_s1_aligned[i-1]  # break below previous day S1
        
        # Return to 1h VWAP for exit
        return_to_vwap = abs(close[i] - vwap[i]) < 0.005 * close[i]  # within 0.5% of VWAP
        
        if position == 0:
            # Long: breakout up + volume filter + price above EMA
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.20
                position = 1
            # Short: breakout down + volume filter + price below EMA
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: return to VWAP or break down
            if return_to_vwap or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: return to VWAP or break up
            if return_to_vwap or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0