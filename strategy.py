#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla R3/S3 breakout with volume confirmation and ATR trailing stop.
Long when price breaks above 1d Camarilla R3 AND volume > 1.5x 20-period average.
Short when price breaks below 1d Camarilla S3 AND volume > 1.5x 20-period average.
Exit when price retraces to 1d Camarilla midpoint (R3/S3 average) or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
Designed for 4h timeframe targeting ~25-35 trades/year per symbol (100-140 total over 4 years).
Focus on BTC and ETH as primary targets with volume confirmation to filter false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2, M = (R3+S3)/2 = C
    # Actually, standard Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use R3 and S3 as the key levels
    range_1d = h_1d - l_1d
    camarilla_r3 = c_1d + range_1d * 1.1 / 4
    camarilla_s3 = c_1d - range_1d * 1.1 / 4
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2.0  # which equals c_1d
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # vol MA needs 20, Camarilla needs 1 day
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        mid_val = camarilla_mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R3 AND volume spike
            if price > r3_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below 1d Camarilla S3 AND volume spike
            elif price < s3_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 1d Camarilla midpoint
            if position == 1 and price <= mid_val:
                exit_signal = True
            elif position == -1 and price >= mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1d_VolumeSpike_ATRTrailingStop_MidpointExit"
timeframe = "4h"
leverage = 1.0