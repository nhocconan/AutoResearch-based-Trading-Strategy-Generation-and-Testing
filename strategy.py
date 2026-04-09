#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ehlers Fisher Transform with volume confirmation and regime filter
# Fisher Transform identifies extreme price movements likely to reverse. Works in both bull and bear markets
# Long when Fisher crosses above -1.5 from below, short when crosses below +1.5 from above
# Volume confirmation (current 6h volume > 1.5x 20-period average) filters low-conviction signals
# Regime filter: only trade when 6h ADX > 25 (trending market) to avoid chop
# Position size fixed at 0.25 to minimize fee churn
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_fisher_volume_adx_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ehlers Fisher Transform (period=10)
    # Price = (High + Low) / 2
    hl2_1d = (high_1d + low_1d) / 2.0
    
    # Normalize hl2 to [-1, 1] range over lookback period
    max_hl2 = pd.Series(hl2_1d).rolling(window=10, min_periods=10).max().values
    min_hl2 = pd.Series(hl2_1d).rolling(window=10, min_periods=10).min().values
    range_hl2 = max_hl2 - min_hl2
    # Avoid division by zero
    range_hl2 = np.where(range_hl2 == 0, 1, range_hl2)
    value = 2 * ((hl2_1d - min_hl2) / range_hl2 - 0.5)
    # Clamp to [-0.999, 0.999] for Fisher transform
    value = np.clip(value, -0.999, 0.999)
    # Fisher Transform
    fisher_1d = 0.5 * np.log((1 + value) / (1 - value))
    # Smoothed Fisher (signal line)
    fisher_smooth_1d = pd.Series(fisher_1d).ewm(span=3, adjust=False).mean().values
    
    # Calculate 1d ADX (14-period) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di_14 = np.where(tr_14 != 0, 100 * minus_dm_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Fisher, smoothed Fisher, and ADX to 6h timeframe
    fisher_aligned = align_htf_to_ltf(prices, df_1d, fisher_1d)
    fisher_smooth_aligned = align_htf_to_ltf(prices, df_1d, fisher_smooth_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(fisher_aligned[i]) or np.isnan(fisher_smooth_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        regime_filter = adx_aligned[i] > 25
        
        if not (volume_confirmed and regime_filter):
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Fisher crosses below smoothed Fisher (mean reversion)
            if fisher_aligned[i] < fisher_smooth_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Fisher crosses above smoothed Fisher (mean reversion)
            if fisher_aligned[i] > fisher_smooth_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Fisher Transform signals with volume and regime confirmation
            # Long when Fisher crosses above -1.5 from below
            if (fisher_aligned[i] > -1.5 and fisher_smooth_aligned[i] <= -1.5 and
                fisher_aligned[i-1] <= -1.5 and fisher_smooth_aligned[i-1] > -1.5):
                position = 1
                signals[i] = position_size
            # Short when Fisher crosses below +1.5 from above
            elif (fisher_aligned[i] < 1.5 and fisher_smooth_aligned[i] >= 1.5 and
                  fisher_aligned[i-1] >= 1.5 and fisher_smooth_aligned[i-1] < 1.5):
                position = -1
                signals[i] = -position_size
    
    return signals