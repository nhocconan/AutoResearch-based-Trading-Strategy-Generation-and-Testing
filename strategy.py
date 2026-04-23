#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 12h ADX trend filter + volume confirmation.
Long when Williams %R < -80 (oversold) AND 12h ADX > 25 (trending) AND volume > 1.5x average.
Short when Williams %R > -20 (overbought) AND 12h ADX > 25 (trending) AND volume > 1.5x average.
Exit when Williams %R reverts to -50 (mean reversion) OR ADX < 20 (trend weakens).
Uses 6h timeframe to target ~12-30 trades/year, minimizing fee drag while capturing mean reversion in trending markets.
Works in both bull and bear markets by requiring ADX > 25 for trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Williams %R - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R for 6h timeframe (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 12h data for ADX - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX for 12h timeframe (14-period)
    # ADX calculation requires +DM, -DM, TR
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # Invert to get positive values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = np.abs(np.diff(high_12h, prepend=high_12h[0]))
    tr2 = np.abs(np.diff(low_12h, prepend=low_12h[0]))
    tr3 = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smoothed values using Wilder's smoothing (EMA-like with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed values
    atr = np.zeros_like(true_range)
    atr[0] = true_range[0]
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    
    # Wilder's smoothing
    for i in range(1, len(true_range)):
        atr[i] = atr[i-1] + (alpha * (true_range[i] - atr[i-1]))
        plus_dm_smooth[i] = plus_dm_smooth[i-1] + (alpha * (plus_dm[i] - plus_dm_smooth[i-1]))
        minus_dm_smooth[i] = minus_dm_smooth[i-1] + (alpha * (minus_dm[i] - minus_dm_smooth[i-1]))
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    
    # ADX is smoothed DX
    adx = np.zeros_like(dx)
    adx[period-1] = dx[period-1]  # Seed value
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND ADX > 25 (trending) AND volume spike
            if (wr_val < -80 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND ADX > 25 (trending) AND volume spike
            elif (wr_val > -20 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R reverts to -50 OR ADX < 20 (trend weakens)
                if wr_val >= -50 or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R reverts to -50 OR ADX < 20 (trend weakens)
                if wr_val <= -50 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_ADX_Volume_MeanReversion"
timeframe = "6h"
leverage = 1.0