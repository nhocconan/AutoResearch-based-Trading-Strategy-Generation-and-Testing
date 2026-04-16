#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d pivot continuation and volume confirmation.
# Long when price breaks above 20-period 6h high AND closes above 1d R1 pivot AND volume > 1.3x 20-period 6h average.
# Short when price breaks below 20-period 6h low AND closes below 1d S1 pivot AND volume > 1.3x 20-period 6h average.
# Uses discrete position size 0.25. Designed to capture breakouts with institutional pivot level confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    vol_6h = df_6h['volume'].values
    
    # Donchian upper and lower bands (20-period)
    dc_upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    dc_lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    dc_mid_6h = (dc_upper_6h + dc_lower_6h) / 2
    
    # Volume average (20-period)
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align 6h indicators to primary timeframe (6h)
    dc_upper_6h_aligned = align_htf_to_ltf(prices, df_6h, dc_upper_6h)
    dc_lower_6h_aligned = align_htf_to_ltf(prices, df_6h, dc_lower_6h)
    dc_mid_6h_aligned = align_htf_to_ltf(prices, df_6h, dc_mid_6h)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # === 1d Indicators: Pivot Points (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Classic pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align 1d pivot points to primary timeframe (6h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian/volume)
    warmup = 20
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(dc_upper_6h_aligned[i]) or np.isnan(dc_lower_6h_aligned[i]) or np.isnan(dc_mid_6h_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_ma = vol_ma_6h_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < dc_mid_6h_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > dc_mid_6h_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.3x 20-period average
            volume_confirm = volume[i] > (1.3 * vol_ma)
            
            # LONG: Price breaks above Donchian upper AND closes above 1d R1 pivot AND volume confirmation
            if price > dc_upper_6h_aligned[i] and close[i] > r1 and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND closes below 1d S1 pivot AND volume confirmation
            elif price < dc_lower_6h_aligned[i] and close[i] < s1 and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1dPivotR1S1_VolumeConfirm_V1"
timeframe = "6h"
leverage = 1.0