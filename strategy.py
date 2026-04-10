#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) + 1d volume spike + 1w ADX regime filter
# - Primary signal: Williams %R crosses above -50 (bullish) or below -50 (bearish) on 6h
# - Volume confirmation: 1d volume > 1.5x 20-period average volume (avoid low-participation signals)
# - Regime filter: 1w ADX > 25 (trending market) enables Williams %R cross signals
# - Works in bull/bear: In trending markets (ADX > 25), Williams %R crosses capture momentum;
#   in ranging markets (ADX <= 25), no signals are generated to avoid whipsaw
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1d_1w_williamsr_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_rolled = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_rolled = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_rolled = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    tr_rolled = np.where(tr_rolled == 0, 1e-10, tr_rolled)
    
    plus_di = 100 * plus_dm_rolled / tr_rolled
    minus_di = 100 * minus_dm_rolled / tr_rolled
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    
    williams_r = -100 * (highest_high - close_6h) / hh_ll
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses back below -50
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses back above -50
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R crosses with volume spike and ADX regime filter
            # Only trade in trending markets (ADX > 25)
            if volume_spike_aligned[i] and adx_aligned[i] > 25:
                # Long: Williams %R crosses above -50 (from below)
                if williams_r[i] >= -50 and williams_r[i-1] < -50:
                    position = 1
                    signals[i] = 0.25
                # Short: Williams %R crosses below -50 (from above)
                elif williams_r[i] <= -50 and williams_r[i-1] > -50:
                    position = -1
                    signals[i] = -0.25
    
    return signals