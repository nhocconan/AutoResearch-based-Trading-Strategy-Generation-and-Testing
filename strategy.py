#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d Williams Alligator trend filter and volume confirmation.
# Williams Alligator (JAW=TEETH=LIPS) uses SMAs to identify trending vs ranging markets.
# Long when price breaks above R3 + Alligator aligned bullish (JAW>TEETH>LIPS) + volume > 1.8x 20-bar average.
# Short when price breaks below S3 + Alligator aligned bearish (JAW<TEETH<LIPS) + volume > 1.8x 20-bar average.
# ATR trailing stop (2.0x) for risk management.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).
# Uses 1d HTF for Alligator trend filter (more stable than lower TFs) and volume confirmation to reduce false breakouts.
# Camarilla pivot levels from 1d provide institutional structure; breakouts with volume confirm conviction.

name = "6h_Camarilla_R3S3_1dAlligator_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams Alligator and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator: JAW (13), TEETH (8), LIPS (5) SMAs
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily candle)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    camarilla_r3 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d_vals - (high_1d - low_1d) * 1.1 / 2
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 13  # warmup for Alligator (JAW needs 13)
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: Williams Alligator alignment
        is_alligator_bullish = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        is_alligator_bearish = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 + Alligator bullish + volume confirmation
            if curr_high > r3_aligned[i] and is_alligator_bullish and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below S3 + Alligator bearish + volume confirmation
            elif curr_low < s3_aligned[i] and is_alligator_bearish and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals