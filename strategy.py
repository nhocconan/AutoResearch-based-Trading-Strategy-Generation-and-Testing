#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v3
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Designed for low trade frequency (~25-35/year) to minimize fee drag. Uses 4h primary timeframe with 1d HTF for trend and volume context.
Camarilla levels derived from prior 1d OHLC provide institutional reference points that work across regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Calculate Camarilla levels from prior 1d bar ===
    # Prior day OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2, R3, S3, R4, S4
    # R4 = Close + (High-Low) * 1.1/2
    # R3 = Close + (High-Low) * 1.1/4
    # R2 = Close + (High-Low) * 1.1/6
    # R1 = Close + (High-Low) * 1.1/12
    # S1 = Close - (High-Low) * 1.1/12
    # S2 = Close - (High-Low) * 1.1/6
    # S3 = Close - (High-Low) * 1.1/4
    # S4 = Close - (High-Low) * 1.1/2
    
    rng = (high_1d - low_1d) * 1.1
    camarilla_r1 = close_1d_prev + rng / 12
    camarilla_s1 = close_1d_prev - rng / 12
    camarilla_r2 = close_1d_prev + rng / 6
    camarilla_s2 = close_1d_prev - rng / 6
    camarilla_r3 = close_1d_prev + rng / 4
    camarilla_s3 = close_1d_prev - rng / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_s2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above 1d EMA34
            if price_close > r1 and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below 1d EMA34
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price drops below S1 or reverses below R1 with weakening volume
            if price_close < s1 or (price_close < r1 and vol_spike < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 or reverses above S1 with weakening volume
            if price_close > r1 or (price_close > s1 and vol_spike < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v3"
timeframe = "4h"
leverage = 1.0