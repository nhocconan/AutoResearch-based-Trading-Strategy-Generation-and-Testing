#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with volume spike and Bollinger Band squeeze filter
# Combines Camarilla breakout logic with volatility contraction/expansion pattern.
# Bollinger Band squeeze (low volatility) precedes breakouts; volume spike confirms conviction.
# Weekly trend filter ensures trades align with higher timeframe momentum.
# Designed for 6h timeframe to achieve 12-37 trades/year with controlled risk.

name = "6h_Camarilla_R3_S3_Breakout_VolumeSpike_BBSqueeze_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_ma = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2.0 * bb_std
    bb_lower = bb_ma - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma  # Normalized width
    
    # Bollinger Band squeeze: width < 20th percentile (low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.2).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Calculate Camarilla pivots from previous day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align daily indicators to 6h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 20)  # Bollinger Bands, weekly EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(bb_squeeze_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla breakout + BB squeeze + volume confirm + weekly trend alignment
        vol_confirm = volume_confirm[i]
        bb_squeeze_active = bb_squeeze_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Camarilla R3, above weekly EMA20, BB squeeze, volume confirm
            if (price > camarilla_r3_aligned[i] and price > ema_20_1w_aligned[i] and 
                bb_squeeze_active and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Price < Camarilla S3, below weekly EMA20, BB squeeze, volume confirm
            elif (price < camarilla_s3_aligned[i] and price < ema_20_1w_aligned[i] and 
                  bb_squeeze_active and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to weekly EMA20 or below Camarilla S3
            if price < ema_20_1w_aligned[i] or price < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to weekly EMA20 or above Camarilla R3
            if price > ema_20_1w_aligned[i] or price > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals