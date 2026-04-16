#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation.
# Long when price breaks above R4 with volume > 1.5x 20-bar average.
# Short when price breaks below S4 with volume > 1.5x 20-bar average.
# Exit when price returns to the 1d VWAP (mean reversion to daily fair value).
# Uses discrete position size 0.25. Camarilla R4/S4 represent strong breakout levels.
# Volume confirmation ensures breakouts are genuine, not false moves.
# 1d VWAP exit provides logical profit target in ranging markets.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns).
# 6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R4, S4) and VWAP ===
    # Pivot point = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # S4 = Close - Range * 1.1/2
    # VWAP = sum(Price * Volume) / sum(Volume)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Calculate 1d VWAP (typical price * volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.cumsum(typical_price_1d * volume_1d) / np.cumsum(volume_1d)
    # Handle division by zero at start
    vwap_1d = np.where(np.cumsum(volume_1d) == 0, typical_price_1d, vwap_1d)
    
    # Align all indicators to primary timeframe (6h)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: 20-bar average volume on 6h chart
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # 20 for volume SMA + buffer
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        vwap = vwap_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_sma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_confirmed = vol > (1.5 * vol_ma)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to VWAP (mean reversion)
            if price <= vwap:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP (mean reversion)
            if price >= vwap:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R4 with volume confirmation
            if (price > r4) and vol_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S4 with volume confirmation
            elif (price < s4) and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dCamarilla_R4S4_Breakout_Volume_VWAPExit_V1"
timeframe = "6h"
leverage = 1.0