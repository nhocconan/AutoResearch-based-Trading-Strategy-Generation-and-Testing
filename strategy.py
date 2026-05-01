#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA20 trend filter and volume spike confirmation
# Uses Camarilla pivot levels (R3/S3) from 4h for significant support/resistance, filtered by 4h EMA trend.
# Volume spike ensures institutional participation. Designed to capture strong momentum moves while avoiding chop.
# Uses 1h only for entry timing precision, signal direction from 4h/1d.
# Target: 15-35 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

name = "1h_Camarilla_R3S3_Breakout_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for Camarilla pivots (R3, S3) and EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R3 = H + 1.1*(H-L)/2, S3 = L - 1.1*(H-L)/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r3 = high_4h + 1.1 * (high_4h - low_4h) / 2.0
    camarilla_s3 = low_4h - 1.1 * (high_4h - low_4h) / 2.0
    
    # Align Camarilla levels to 1h timeframe (using previous 4h bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 30  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 4h EMA20
        uptrend = curr_close > ema_20_4h_aligned[i]
        downtrend = curr_close < ema_20_4h_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_close > camarilla_r3_aligned[i]  # Break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below S3 or trend reversal
            if curr_close < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on break above R3 or trend reversal
            if curr_close > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals