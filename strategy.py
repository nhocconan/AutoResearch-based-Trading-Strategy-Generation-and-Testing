#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d volume spike + 1w trend filter (ADX)
# - Primary signal: Williams %R(14) on 6h crosses below -80 (oversold) for long, above -20 (overbought) for short
# - Volume confirmation: 1d volume > 1.3x 20-period average volume (ensure participation)
# - Trend filter: 1w ADX(14) > 25 to trade only in trending markets, avoiding whipsaws in ranges
# - Works in bull/bear: In trends (ADX > 25), momentum extremes precede continuations; avoids false signals in chop
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
    volume_spike = volume_1d > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    # Smoothed TR, +DM, -DM
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    up_sum = pd.Series(up_move).rolling(window=14, min_periods=14).sum().values
    down_sum = pd.Series(down_move).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
    
    # DI+ and DI-
    di_plus = 100 * up_sum / tr_sum_safe
    di_minus = 100 * down_sum / tr_sum_safe
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0,
                          ((highest_high - close_6h) / denominator) * -100,
                          -50)  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (momentum fading) or reverse signal
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (momentum fading) or reverse signal
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume spike and trend filter
            # Only trade when ADX > 25 (trending market)
            if adx_aligned[i] > 25 and volume_spike_aligned[i]:
                # Long: Williams %R crosses below -80 (oversold)
                if williams_r[i] <= -80 and williams_r[i-1] > -80:
                    position = 1
                    signals[i] = 0.25
                # Short: Williams %R crosses above -20 (overbought)
                elif williams_r[i] >= -20 and williams_r[i-1] < -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals