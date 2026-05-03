#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R(14) with 1d ADX trend filter and volume confirmation.
# Long: %R crosses above -80 from below AND 1d ADX > 25 AND volume > 1.5x 20-period MA
# Short: %R crosses below -20 from above AND 1d ADX > 25 AND volume > 1.5x 20-period MA
# Exit: Opposite %R cross or ADX < 20 (range) or volume drops.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Williams %R identifies overbought/oversold conditions; 1d ADX filters for trending markets only;
# volume confirmation reduces false signals. Works in bull via long signals from oversold
# and bear via short signals from overbought when aligned with higher timeframe trend.

name = "4h_WilliamsR14_1dADX25_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low), 
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)), 
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R (14-period) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Determine trend regime
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Williams %R cross signals
        cross_above_80 = wr > -80 and wr_prev <= -80
        cross_below_20 = wr < -20 and wr_prev >= -20
        
        # Entry logic
        if position == 0:
            # Long: %R crosses above -80 from below AND trending AND volume spike
            if cross_above_80 and is_trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: %R crosses below -20 from above AND trending AND volume spike
            elif cross_below_20 and is_trending and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: %R crosses below -20 from above OR trend weakens (ADX < 20) OR volume drops
            if cross_below_20 or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %R crosses above -80 from below OR trend weakens (ADX < 20) OR volume drops
            if cross_above_80 or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals