#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Camarilla pivot levels (R1/S1) with 1d volume spike filter and ATR-based stoploss.
# Long when price breaks above weekly Camarilla R1 with 1d volume > 1.5x 20-period average.
# Short when price breaks below weekly Camarilla S1 with 1d volume > 1.5x 20-period average.
# Exit via ATR trailing stop: signal=0 when long position price < highest_high - 2.0*ATR or short position price > lowest_low + 2.0*ATR.
# Uses discrete position size 0.25. Weekly Camarilla provides structure from higher timeframe, 12h provides entry timing with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing meaningful breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: Camarilla Pivot Levels (R1, S1) based on prior week ===
    # Calculate using prior week's high, low, close (shift by 1 to use completed week only)
    phigh = np.roll(high_1w, 1)
    plow = np.roll(low_1w, 1)
    pclose = np.roll(close_1w, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Camarilla levels (based on prior week)
    pivot = (phigh + plow + pclose) / 3.0
    camarilla_r1 = pivot + (1.1/12) * (phigh - plow)  # R1 level
    camarilla_s1 = pivot - (1.1/12) * (phigh - plow)  # S1 level
    
    # Align weekly Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Get daily data once before loop for volume filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Daily Indicators: Volume moving average (20-period) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === Daily Indicators: ATR (14) for trailing stop ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state, entry price, and highest/lowest since entry for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values
        cr1 = camarilla_r1_aligned[i]
        cs1 = camarilla_s1_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        atr_val = atr_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC: ATR Trailing Stop ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update highest since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit if price drops below highest_since_entry - 2.0 * ATR
            if price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update lowest since entry
            if price < lowest_since_entry or lowest_since_entry == 0.0:
                lowest_since_entry = price
            # Exit if price rises above lowest_since_entry + 2.0 * ATR
            if price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Price breaks above weekly Camarilla R1 with volume confirmation
            if (price > cr1) and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                lowest_since_entry = 0.0  # reset for short
            
            # SHORT: Price breaks below weekly Camarilla S1 with volume confirmation
            elif (price < cs1) and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                highest_since_entry = 0.0  # reset for long
        
        else:
            # Maintain current position size
            signals[i] = position * 0.25
    
    return signals

name = "12h_1wCamarillaR1S1_1dVolumeSpike_ATRTrailingStop_V1"
timeframe = "12h"
leverage = 1.0