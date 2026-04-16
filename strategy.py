#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation.
# Long when price breaks above R4 with volume > 1.5x 20-period average.
# Short when price breaks below S4 with volume > 1.5x 20-period average.
# Exit when price returns to the 1d VWAP (mean reversion to daily fair value).
# Uses discrete position size 0.25. Camarilla levels provide institutional support/resistance,
# volume confirms institutional participation, VWAP exit provides mean reversion edge.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R1-R4, S1-S4) and VWAP ===
    # Pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_1d * 1.0 / 12)
    r2 = pp + (range_1d * 2.0 / 12)
    r3 = pp + (range_1d * 3.0 / 12)
    r4 = pp + (range_1d * 4.0 / 12)
    s1 = pp - (range_1d * 1.0 / 12)
    s2 = pp - (range_1d * 2.0 / 12)
    s3 = pp - (range_1d * 3.0 / 12)
    s4 = pp - (range_1d * 4.0 / 12)
    
    # 1d VWAP (for exit)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vp_1d = typical_price_1d * volume_1d
    cum_vp_1d = np.cumsum(vp_1d)
    cum_vol_1d = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_vp_1d, cum_vol_1d, out=np.zeros_like(cum_vp_1d), where=cum_vol_1d!=0)
    
    # Align 1d indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Get 6h data for volume MA (volume confirmation)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # Volume moving average (20-period) on 6h
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        vwap_val = vwap_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to 1d VWAP (mean reversion)
            if price <= vwap_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to 1d VWAP (mean reversion)
            if price >= vwap_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average (6h)
            vol_filter = vol > 1.5 * vol_ma_val
            
            # LONG: price breaks above R4 with volume confirmation
            if price > r4_val and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below S4 with volume confirmation
            elif price < s4_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dCamarillaR4S4_VolumeConfirmation_VWAPExit_V1"
timeframe = "6h"
leverage = 1.0