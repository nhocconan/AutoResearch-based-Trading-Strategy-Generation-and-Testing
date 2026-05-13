#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 levels act as strong support/resistance. Breaks above R3 or below S3 with volume confirmation and aligned 1d trend (via Supertrend) capture momentum moves. Works in bull markets by catching breakouts and in bear markets by catching sharp reversals. Uses discrete position sizing to limit turnover and uses volume spike to avoid false breakouts.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla levels
    typical_price = (high + low + close) / 3
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's data for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day: use same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        range_ = prev_high[i] - prev_low[i]
        camarilla_r3[i] = prev_close[i] + range_ * 1.1 / 4
        camarilla_s3[i] = prev_close[i] - range_ * 1.1 / 4
        camarilla_r4[i] = prev_close[i] + range_ * 1.1 / 2
        camarilla_s4[i] = prev_close[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d Supertrend for trend filter
    atr_period = 10
    atr_multiplier = 3.0
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    atr_1d = np.zeros_like(close_1d)
    atr_1d[:atr_period] = np.mean(tr_1d[:atr_period])
    for i in range(atr_period, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * (atr_period-1) + tr_1d[i]) / atr_period
    
    hl2_1d = (high_1d + low_1d) / 2
    upper_band_1d = hl2_1d + (atr_multiplier * atr_1d)
    lower_band_1d = hl2_1d - (atr_multiplier * atr_1d)
    
    supertrend_1d = np.ones_like(close_1d)
    for i in range(1, len(close_1d)):
        if close_1d[i] > upper_band_1d[i-1]:
            supertrend_1d[i] = 1
        elif close_1d[i] < lower_band_1d[i-1]:
            supertrend_1d[i] = -1
        else:
            supertrend_1d[i] = supertrend_1d[i-1]
            if supertrend_1d[i] == 1 and lower_band_1d[i] < lower_band_1d[i-1]:
                lower_band_1d[i] = lower_band_1d[i-1]
            if supertrend_1d[i] == -1 and upper_band_1d[i] > upper_band_1d[i-1]:
                upper_band_1d[i] = upper_band_1d[i-1]
    
    # Align 1d Supertrend to 6h timeframe
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    
    # Calculate volume average (24-period) for volume spike filter
    vol_ma_24 = np.zeros_like(volume)
    for i in range(23, len(volume)):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(supertrend_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 24-period average
        vol_spike = volume[i] > 2.0 * vol_ma_24[i]
        
        if position == 0:
            # LONG: Close above R3 with volume spike and 1d uptrend
            if (close[i] > r3_6h[i] and vol_spike and 
                supertrend_1d_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 with volume spike and 1d downtrend
            elif (close[i] < s3_6h[i] and vol_spike and 
                  supertrend_1d_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below R3 or trend reversal
            if (close[i] < r3_6h[i] or supertrend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above S3 or trend reversal
            if (close[i] > s3_6h[i] or supertrend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals