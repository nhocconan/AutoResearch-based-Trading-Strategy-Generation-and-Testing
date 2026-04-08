#!/usr/bin/env python3
# 6h Weekly Pivot Reversal with Volume Confirmation and ADX Trend Filter
# Hypothesis: Weekly pivot points act as major institutional support/resistance.
# Price rejecting at weekly R3/S3 with volume > 1.5x 20-period average and ADX > 25 indicates institutional defense of levels.
# Works in bull/bear markets by capturing reversals from key weekly levels with trend filter to avoid false signals in weak trends.
# Target: 15-35 trades/year per symbol.

name = "6h_weekly_pivot_reversal_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and ADX - call ONCE before loop
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    volume_w = df_w['volume'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pp_w = (high_w + low_w + close_w) / 3
    r3_w = pp_w + 2 * (high_w - low_w)  # R3 = P + 2*(H-L)
    s3_w = pp_w - 2 * (high_w - low_w)  # S3 = P - 2*(H-L)
    
    # Calculate 20-period average volume for weekly timeframe
    vol_ma_w = pd.Series(volume_w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on weekly timeframe (trend strength filter)
    # True Range
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_w - np.roll(high_w, 1)
    down_move = np.roll(low_w, 1) - low_w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Get aligned weekly values for current 6h bar
        r3 = align_htf_to_ltf(prices, df_w, r3_w)[i]
        s3 = align_htf_to_ltf(prices, df_w, s3_w)[i]
        vol_ma = align_htf_to_ltf(prices, df_w, vol_ma_w)[i]
        adx_val = align_htf_to_ltf(prices, df_w, adx)[i]
        
        # Skip if any required data is NaN
        if np.isnan(r3) or np.isnan(s3) or np.isnan(vol_ma) or np.isnan(adx_val) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume rejection condition: current volume > 1.5x 20-period average
        vol_rejection = volume[i] > 1.5 * vol_ma
        
        # Strong trend condition: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 1:  # Long position
            # Exit if price breaks below S3 (rejection failed)
            if close[i] < s3:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above R3 (rejection failed)
            if close[i] > r3:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Rejection long at S3 with volume confirmation and strong trend
            if low[i] <= s3 and close[i] > s3 and vol_rejection and strong_trend:
                position = 1
                signals[i] = 0.25
            # Rejection short at R3 with volume confirmation and strong trend
            elif high[i] >= r3 and close[i] < r3 and vol_rejection and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals