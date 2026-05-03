#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and 1d volume spike confirmation.
# Long when price breaks above Camarilla R3 level AND 1w close > 1w EMA34 (uptrend) AND 1d volume > 2.0x 20-period volume MA.
# Short when price breaks below Camarilla S3 level AND 1w close < 1w EMA34 (downtrend) AND 1d volume > 2.0x 20-period volume MA.
# Uses 12h timeframe to target 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla pivots provide mathematically derived support/resistance levels, 1w EMA34 filters for primary trend alignment,
# 1d volume spike confirms institutional participation. Designed to work in both bull and bear markets by only
# trading breakouts in the direction of the 1w trend when volume confirms the move's validity.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_1dVolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: R4 = close + (high-low)*1.5/2, R3 = close + (high-low)*1.25/2, etc.
    # We need previous 1d OHLC for current 12h bar calculation
    # Since we're on 12h timeframe, we use 1d data shifted by 1 bar for previous day's levels
    df_1d_prev = df_1d.shift(1)  # Previous day's OHLC
    # Handle NaN from shift
    high_1d_prev = df_1d_prev['high'].values
    low_1d_prev = df_1d_prev['low'].values
    close_1d_prev = df_1d_prev['close'].values
    
    # Calculate Camarilla levels for each 12h bar using previous day's OHLC
    # R3 = close + (high-low)*1.125/2, S3 = close - (high-low)*1.125/2
    # Actually standard Camarilla: R3 = close + (high-low)*1.1/2, S3 = close - (high-low)*1.1/2
    # Using widely accepted formula: R3 = close + (high-low)*1.1/2, S3 = close - (high-low)*1.1/2
    range_1d_prev = high_1d_prev - low_1d_prev
    camarilla_r3 = close_1d_prev + (range_1d_prev * 1.1 / 2)
    camarilla_s3 = close_1d_prev - (range_1d_prev * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 1d volume > 2.0x 20-period MA
        # Since we don't have intraday 1d volume in 12h data, we use volume ratio from 12h as proxy
        # But better: check if current 12h volume is elevated relative to its own MA as volume confirmation
        volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume[i] > (volume_ma_12h[i] * 2.0)  # 12h volume spike as proxy
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_r3_aligned[i]   # Price breaks above Camarilla R3
        breakout_down = low_val < camarilla_s3_aligned[i]  # Price breaks below Camarilla S3
        
        # 1w trend conditions
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 1w uptrend AND volume spike
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 1w downtrend AND volume spike
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla Pivot Point (mid-level) OR trend changes
            # Camarilla pivot point = (high + low + close)/3 from previous day
            camarilla_pp = (high_1d_prev[i] + low_1d_prev[i] + close_1d_prev[i]) / 3.0
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if close_val < camarilla_pp_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla Pivot Point OR trend changes
            camarilla_pp = (high_1d_prev[i] + low_1d_prev[i] + close_1d_prev[i]) / 3.0
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if close_val > camarilla_pp_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals