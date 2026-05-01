#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla R3/S3 from weekly timeframe represent stronger support/resistance levels.
# Breakout above weekly R3 or below weekly S3 with volume spike indicates strong momentum aligned with weekly trend.
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades.
# Volume confirmation filters out false breakouts.
# Designed for fewer, higher-quality trades (target: 12-37/year) to minimize fee drag on 12h timeframe.
# Works in both bull and bear markets by following weekly trend with momentum confirmation.

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w data for Camarilla pivot calculation (previous week's OHLC)
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1w bar
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Camarilla levels: R3 and S3 (stronger breakout signals)
    # R3 = Close + 1.1*(High-Low)/4
    # S3 = Close - 1.1*(High-Low)/4
    camarilla_range = high_1w - low_1w
    r3 = close_1w + 1.1 * camarilla_range / 4
    s3 = close_1w - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 12h timeframe (use previous week's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: current volume > 2.0 * 30-period average volume
    volume_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (volume_ma_30 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above weekly R3, volume spike, uptrend
            if close[i] > r3_aligned[i] and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S3, volume spike, downtrend
            elif close[i] < s3_aligned[i] and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below weekly R3 or trend reversal
            if close[i] < r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above weekly S3 or trend reversal
            if close[i] > s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals