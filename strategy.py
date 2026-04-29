#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions
# ADX > 25 filters for trending markets (avoid whipsaws in ranging markets)
# Volume spike > 2x average confirms institutional participation
# Long: Williams %R < -80 (oversold) AND ADX > 25 AND volume > 2x avg
# Short: Williams %R > -20 (overbought) AND ADX > 25 AND volume > 2x avg
# Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via ADX trend filter.
# Timeframe: 6h (primary), HTF: 1d for ADX trend.

name = "6h_WilliamsR_1dADX_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d ADX trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # +DM = max(high - high_prev, 0) if > max(low_prev - low, 0) else 0
    # -DM = max(low_prev - low, 0) if > max(high - high_prev, 0) else 0
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr)
    
    # DX = |+DI - -DI| / (+DI + -DI) * 100
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    # ADX = smoothed DX
    adx_14_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # warmup for Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(adx_14_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_adx = adx_14_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Williams %R rises above -50 (exiting oversold)
            # 2. ADX falls below 20 (trend weakening)
            if (curr_williams_r > -50 or
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Williams %R falls below -50 (exiting overbought)
            # 2. ADX falls below 20 (trend weakening)
            if (curr_williams_r < -50 or
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume confirm
            if (curr_williams_r < -80 and
                curr_adx > 25 and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume confirm
            elif (curr_williams_r > -20 and
                  curr_adx > 25 and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals