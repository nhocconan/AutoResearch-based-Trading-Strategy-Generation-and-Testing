#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Vortex_Trend_With_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate +DM and -DM for 1d
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing (using EMA as approximation) for TR, +DM, -DM
    alpha = 1.0 / 14
    tr_smooth = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initialize first values
    tr_smooth[13] = np.nansum(tr[1:15])  # Simple sum for first 14 periods
    plus_dm_smooth[13] = np.nansum(plus_dm[1:15])
    minus_dm_smooth[13] = np.nansum(minus_dm[1:15])
    
    # Wilder smoothing for remaining periods
    for i in range(14, len(tr)):
        if np.isnan(tr[i]):
            tr_smooth[i] = tr_smooth[i-1]
            plus_dm_smooth[i] = plus_dm_smooth[i-1]
            minus_dm_smooth[i] = minus_dm_smooth[i-1]
        else:
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
    
    # Calculate Vortex Indicator VI+ and VI-
    vi_plus = plus_dm_smooth / tr_smooth
    vi_minus = minus_dm_smooth / tr_smooth
    
    # Align Vortex indicators to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Volume filter: current volume > 1.5x 24-period average (12h * 2 = 24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: VI+ > VI- with volume confirmation
            if vi_plus_val > vi_minus_val and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ with volume confirmation
            elif vi_minus_val > vi_plus_val and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: VI- crosses above VI+
            if vi_minus_val > vi_plus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: VI+ crosses above VI-
            if vi_plus_val > vi_minus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals