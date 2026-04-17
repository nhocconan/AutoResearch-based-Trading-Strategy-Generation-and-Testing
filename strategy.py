#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Volume_Regime
Strategy: 12-hour breakout of daily Camarilla R1/S1 levels with volume confirmation and 1d trend filter.
Long: Price breaks above daily R1 + volume > 1.8x 20-period avg + price above 1d EMA50
Short: Price breaks below daily S1 + volume > 1.8x 20-period avg + price below 1d EMA50
Exit: Price returns to 12h VWAP or breaks opposite Camarilla level
Position size: 0.25
Designed to capture breakouts aligned with daily trend while avoiding range-bound conditions.
Timeframe: 12h
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
    
    # Calculate 12h VWAP for exit
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate 1d OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Camarilla levels: R1, S1
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    
    # Align 1d levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_1d_aligned[i-1]  # break above previous day R1
        breakout_down = close[i] < s1_1d_aligned[i-1]  # break below previous day S1
        
        # Return to 12h VWAP for exit (within 0.5% of VWAP)
        return_to_vwap = abs(close[i] - vwap[i]) < 0.005 * close[i]
        
        # Exit opposite Camarilla level
        exit_long = close[i] < s1_1d_aligned[i]  # price below S1
        exit_short = close[i] > r1_1d_aligned[i]  # price above R1
        
        if position == 0:
            # Long: breakout above R1 + volume filter + price above EMA
            if breakout_up and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + price below EMA
            elif breakout_down and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to VWAP or price below S1
            if return_to_vwap or exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to VWAP or price above R1
            if return_to_vwap or exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0