#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakouts with 1d volume confirmation.
# Long when price breaks above weekly R4 with volume > 1.5x 20-period average.
# Short when price breaks below weekly S4 with volume > 1.5x 20-period average.
# Exit when price re-enters the weekly R3-S3 range (mean reversion zone).
# Uses discrete position size 0.25. Weekly Camarilla provides structure, 1d volume confirms momentum.
# 6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === Weekly Indicators: Camarilla Pivots (based on prior week) ===
    # Camarilla levels calculated from previous week's OHLC
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    hl_range_1w = high_1w - low_1w
    r4_1w = close_1w + 1.5 * hl_range_1w
    r3_1w = close_1w + 1.0 * hl_range_1w
    s3_1w = close_1w - 1.0 * hl_range_1w
    s4_1w = close_1w - 1.5 * hl_range_1w
    
    # === 1d Indicators: Volume Average for confirmation ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (6h)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Sufficient for weekly and daily calculations
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol > 1.5 * vol_ma if vol_ma > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price re-enters below R3 (mean reversion zone)
            if price < r3:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price re-enters above S3 (mean reversion zone)
            if price > s3:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above weekly R4 with volume confirmation
            if (price > r4) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below weekly S4 with volume confirmation
            elif (price < s4) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_WeeklyCamarilla_R4S4_Breakout_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0