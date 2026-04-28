#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with 1d Volume Spike and ADX25 Trend Filter
# Uses weekly Camarilla pivots (R3/S3 for breakout, R4/S4 for acceleration)
# Long when price breaks above weekly R3 with volume > 2x 20-bar average and ADX > 25
# Short when price breaks below weekly S3 with volume > 2x 20-bar average and ADX > 25
# Exits when price returns to weekly pivot center (PP) or ADX drops below 20
# Weekly pivots calculated from prior week's H/L/C, providing structural levels
# Volume spike filters breakout authenticity, ADX ensures trending conditions
# Target: 15-30 trades/year via tight weekly pivot levels reducing false breakouts
# Works in bull/bear by trading breakouts in direction of higher timeframe trend

name = "6h_WeeklyPivot_R3S3_Breakout_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculations
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from prior week's OHLC
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1
    # S3 = PP - (H - L) * 1.1
    # R4 = PP + (H - L) * 1.5
    # S4 = PP - (H - L) * 1.5
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pp_w = (high_w + low_w + close_w) / 3.0
    range_w = high_w - low_w
    r3_w = pp_w + range_w * 1.1
    s3_w = pp_w - range_w * 1.1
    r4_w = pp_w + range_w * 1.5
    s4_w = pp_w - range_w * 1.5
    
    # Get daily data for ADX and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed DM
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Prepend zeros for alignment (since we lost first bar in calculations)
    adx = np.concatenate([np.full(27, np.nan), adx])  # 1 (TR) + 14 (DM smoothing) + 14 (ADX smoothing) - 2
    
    # Align weekly pivots to 6h timeframe
    pp_w_aligned = align_htf_to_ltf(prices, df_w, pp_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        adx_val = adx_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above weekly R3 with volume confirmation and ADX > 25
            if price > r3_w_aligned[i] and vol_conf and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below weekly S3 with volume confirmation and ADX > 25
            elif price < s3_w_aligned[i] and vol_conf and adx_val > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to weekly PP or ADX < 20 (range) or no volume
            if price < pp_w_aligned[i] or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to weekly PP or ADX < 20 (range) or no volume
            if price > pp_w_aligned[i] or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals