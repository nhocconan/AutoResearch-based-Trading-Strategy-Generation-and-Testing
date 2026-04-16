#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with volume confirmation.
# Long when price breaks above R4 with volume > 1.5x 20-period average.
# Short when price breaks below S4 with volume > 1.5x 20-period average.
# Exit when price returns to the weekly pivot point (PP).
# Uses 1w Camarilla levels as structural resistance/support, effective in both bull and bear markets.
# 6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Volume confirmation reduces false breakouts. Works in trending markets (breakouts continue) and 
# ranging markets (mean reversion to PP after false breakouts).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Camarilla Pivot Levels ===
    # Calculated from previous week's OHLC
    # PP = (H + L + C) / 3
    # R4 = PP + ((H - L) * 1.1 / 2)
    # S4 = PP - ((H - L) * 1.1 / 2)
    def calculate_camarilla(high_arr, low_arr, close_arr):
        """Calculate Camarilla pivot levels for each bar"""
        pp = (high_arr + low_arr + close_arr) / 3.0
        r4 = pp + ((high_arr - low_arr) * 1.1 / 2.0)
        s4 = pp - ((high_arr - low_arr) * 1.1 / 2.0)
        return pp, r4, s4
    
    pp_1w, r4_1w, s4_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Align all indicators to primary timeframe (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # 20 for volume MA + buffer for HTF alignment
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        pp = pp_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_average = vol_ma[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to or below weekly pivot point (PP)
            if price <= pp:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to or above weekly pivot point (PP)
            if price >= pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirmed = vol > (1.5 * vol_average)
            
            # LONG: Price breaks above R4 with volume confirmation
            if (price > r4) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S4 with volume confirmation
            elif (price < s4) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1wCamarilla_R4S4_Breakout_VolumeConf_PPExit_V1"
timeframe = "6h"
leverage = 1.0