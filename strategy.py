#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 12h EMA50 trend filter and volume spike confirmation
# Camarilla pivots from 1d: R3/S3 are strong reversal/breakout levels
# Long when price breaks above R3 with volume > 2x 20-period average and price > 12h EMA50 (uptrend)
# Short when price breaks below S3 with volume > 2x 20-period average and price < 12h EMA50 (downtrend)
# Exit when price returns to opposite Camarilla level (R1/S1) or after max 6 bars
# Designed for ~12-25 trades/year on 6h timeframe to minimize fee drag
# Uses 12h trend filter and volume spike to avoid false breakouts in choppy markets

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-calculate Camarilla levels for each 1d bar
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_S3 = np.zeros(len(df_1d))
    camarilla_R1 = np.zeros(len(df_1d))
    camarilla_S1 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        diff = h - l
        camarilla_R3[i] = c + diff * 1.1 / 4
        camarilla_S3[i] = c - diff * 1.1 / 4
        camarilla_R1[i] = c + diff * 1.1 / 12
        camarilla_S1[i] = c - diff * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        curr_R1 = camarilla_R1_aligned[i]
        curr_S1 = camarilla_S1_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle position exits and management
        if position == 1:  # Long position
            bars_in_trade += 1
            # Exit: price returns to R1 or max 6 bars (36h) holding period
            if curr_low <= curr_R1 or bars_in_trade >= 6:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_in_trade += 1
            # Exit: price returns to S1 or max 6 bars (36h) holding period
            if curr_high >= curr_S1 or bars_in_trade >= 6:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 2.0x 20-period average
            vol_spike = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above R3 with volume spike and uptrend (price > 12h EMA50)
            if vol_spike and curr_high > curr_R3 and curr_close > curr_ema50_12h:
                signals[i] = 0.25
                position = 1
                bars_in_trade = 1
            # Short entry: price breaks below S3 with volume spike and downtrend (price < 12h EMA50)
            elif vol_spike and curr_low < curr_S3 and curr_close < curr_ema50_12h:
                signals[i] = -0.25
                position = -1
                bars_in_trade = 1
            else:
                signals[i] = 0.0
    
    return signals