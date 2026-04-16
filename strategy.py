#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla R3/S3 levels with 12h volume spike and 4h choppiness filter.
# Long when price breaks above daily Camarilla R3 with 12h volume > 2.0x 24-period average and CHOP > 61.8 (range).
# Short when price breaks below daily Camarilla S3 with same filters.
# Exit when price returns to daily Camarilla midpoint or touches opposite S3/R3 level.
# Uses discrete position size 0.25. CHOP filter ensures mean-reversion logic in ranging markets.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: Camarilla Pivot Levels (R3, S3, Midpoint) based on prior day ===
    # Calculate using prior day's high, low, close (shift by 1 to use completed day only)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Camarilla levels (based on prior day)
    pivot = (phigh + plow + pclose) / 3.0
    camarilla_r3 = pivot + (1.1/4) * (phigh - plow)  # R3 = pivot + 1.1/4*(H-L)
    camarilla_s3 = pivot - (1.1/4) * (phigh - plow)  # S3 = pivot - 1.1/4*(H-L)
    camarilla_mid = pivot
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Get 12h data once before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Volume Spike (24-period average) ===
    vol_ma_24_12h = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    vol_ma_24_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_24_12h)
    
    # Get 4h data for choppiness filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Choppiness Index (CHOP) = 100 * log10(sum(TR over n) / (n * max(high-low over n))) / log10(n)
    n_chop = 14
    sum_tr = pd.Series(tr).rolling(window=n_chop, min_periods=n_chop).sum().values
    max_hl = pd.Series(high - low).rolling(window=n_chop, min_periods=n_chop).max().values
    chop = 100 * np.log10(sum_tr / (n_chop * max_hl)) / np.log10(n_chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(vol_12h_aligned[i]) or 
            np.isnan(vol_ma_24_12h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        cr3 = camarilla_r3_aligned[i]
        cs3 = camarilla_s3_aligned[i]
        cm = camarilla_mid_aligned[i]
        vol_12h_val = vol_12h_aligned[i]
        vol_ma_24_12h_val = vol_ma_24_12h_aligned[i]
        chop_val = chop[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to daily Camarilla midpoint or drops to Camarilla S3
            if price <= cm or price <= cs3:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to daily Camarilla midpoint or rises to Camarilla R3
            if price >= cm or price >= cr3:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: 12h volume > 2.0x 24-period average
            vol_filter = vol_12h_val > 2.0 * vol_ma_24_12h_val
            
            # Choppiness filter: CHOP > 61.8 indicates ranging market (mean reversion regime)
            chop_filter = chop_val > 61.8
            
            # LONG: Price breaks above daily Camarilla R3 with volume and chop confirmation
            if (price > cr3) and vol_filter and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below daily Camarilla S3 with volume and chop confirmation
            elif (price < cs3) and vol_filter and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dCamarillaR3S3_12hVolSpike_4hChopFilter_V1"
timeframe = "4h"
leverage = 1.0