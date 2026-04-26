#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeRegime
Hypothesis: Trade Camarilla R3/S3 breakouts only when aligned with 1d EMA34 trend and in low-chop regime (BW<50%). Uses volume confirmation (top 30%) to ensure institutional participation. Fixed size 0.25 to limit trades (~25/year). Designed to work in both bull (trend following) and bear (mean reversion to midpoint) markets by exiting at Camarilla center.
"""

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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d_vals)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Width regime filter (20,2) - low chop = BW < 50%
    close_series = pd.Series(close)
    ma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + 2 * std_20
    lower_band = ma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / ma_20 * 100
    low_chop_regime = bb_width < 50.0  # trending market
    
    # Volume spike: volume > 70th percentile of 20-period lookback
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=20, min_periods=20).quantile(0.70).values
    volume_spike = volume > vol_percentile_70
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (34 for EMA, 20 for BB/volume)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(vol_percentile_70[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        low_chop = low_chop_regime[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume spike AND aligned with 1d EMA34 trend AND low chop regime
        long_entry = (close_val > camarilla_r3_val) and vol_spike and (close_val > ema_34_val) and low_chop
        short_entry = (close_val < camarilla_s3_val) and vol_spike and (close_val < ema_34_val) and low_chop
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0