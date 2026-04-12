#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and ADX trend filter
    # Camarilla levels from 1d provide institutional support/resistance for daily entries
    # 1w volume spike confirms institutional participation across the week
    # ADX filter ensures we only trade in trending markets (ADX > 25) to avoid whipsaws
    # Works in bull/bear by following institutional breakouts with volume confirmation
    # Target: 20-50 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (using previous day's range)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_pivot = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        # Pivot point
        camarilla_pivot[i] = (phigh + plow + 2 * pclose) / 4
        
        # Range
        rng = phigh - plow
        
        # Camarilla levels
        camarilla_h4[i] = camarilla_pivot[i] + rng * 1.1 / 2
        camarilla_l4[i] = camarilla_pivot[i] - rng * 1.1 / 2
        camarilla_h3[i] = camarilla_pivot[i] + rng * 1.1 / 4
        camarilla_l3[i] = camarilla_pivot[i] - rng * 1.1 / 4
    
    # Align 1d Camarilla levels to 1d timeframe (same timeframe, so direct alignment)
    # Since we're on 1d timeframe, we can use the values directly with proper shift
    h4_aligned = np.roll(camarilla_h4, 1)
    l4_aligned = np.roll(camarilla_l4, 1)
    h3_aligned = np.roll(camarilla_h3, 1)
    l3_aligned = np.roll(camarilla_l3, 1)
    pivot_aligned = np.roll(camarilla_pivot, 1)
    # Set first value to nan as there's no previous day
    h4_aligned[0] = np.nan
    l4_aligned[0] = np.nan
    h3_aligned[0] = np.nan
    l3_aligned[0] = np.nan
    pivot_aligned[0] = np.nan
    
    # Get 1w data for volume confirmation and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w volume spike filter (current volume > 2.0 * 20-week average)
    vol_ma_20_1w = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        vol_ma_20_1w[i] = np.mean(volume_1w[i-19:i+1])
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    volume_spike = volume > 2.0 * vol_ma_20_1w_aligned
    
    # 1w ADX calculation (14-period)
    # True Range
    tr_1w = np.full(len(df_1w), np.nan)
    for i in range(1, len(df_1w)):
        tr_1w[i] = np.max([
            high_1w[i] - low_1w[i],
            np.abs(high_1w[i] - close_1w[i-1]),
            np.abs(low_1w[i] - close_1w[i-1])
        ])
    
    # Directional Movement
    dm_plus_1w = np.full(len(df_1w), np.nan)
    dm_minus_1w = np.full(len(df_1w), np.nan)
    for i in range(1, len(df_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        dm_plus_1w[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus_1w[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    atr_1w = np.full(len(df_1w), np.nan)
    dm_plus_smooth_1w = np.full(len(df_1w), np.nan)
    dm_minus_smooth_1w = np.full(len(df_1w), np.nan)
    
    # Initial values (first 14 periods)
    for i in range(14, len(df_1w)):
        if i == 14:
            atr_1w[i] = np.sum(tr_1w[1:15])
            dm_plus_smooth_1w[i] = np.sum(dm_plus_1w[1:15])
            dm_minus_smooth_1w[i] = np.sum(dm_minus_1w[1:15])
        else:
            atr_1w[i] = atr_1w[i-1] - (atr_1w[i-1] / 14) + tr_1w[i]
            dm_plus_smooth_1w[i] = dm_plus_smooth_1w[i-1] - (dm_plus_smooth_1w[i-1] / 14) + dm_plus_1w[i]
            dm_minus_smooth_1w[i] = dm_minus_smooth_1w[i-1] - (dm_minus_smooth_1w[i-1] / 14) + dm_minus_1w[i]
    
    # Directional Indicators
    di_plus_1w = np.full(len(df_1w), np.nan)
    di_minus_1w = np.full(len(df_1w), np.nan)
    dx_1w = np.full(len(df_1w), np.nan)
    
    for i in range(14, len(df_1w)):
        if atr_1w[i] > 0:
            di_plus_1w[i] = 100 * dm_plus_smooth_1w[i] / atr_1w[i]
            di_minus_1w[i] = 100 * dm_minus_smooth_1w[i] / atr_1w[i]
            if di_plus_1w[i] + di_minus_1w[i] > 0:
                dx_1w[i] = 100 * np.abs(di_plus_1w[i] - di_minus_1w[i]) / (di_plus_1w[i] + di_minus_1w[i])
    
    # ADX (smoothed DX)
    adx_1w = np.full(len(df_1w), np.nan)
    for i in range(28, len(df_1w)):  # 14 + 14 for smoothing
        if i == 28:
            adx_1w[i] = np.mean(dx_1w[15:29])
        else:
            adx_1w[i] = (adx_1w[i-1] * 13 + dx_1w[i]) / 14
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_1w_aligned[i] > 25:
            # Breakout entries with volume confirmation
            long_entry = close[i] > h4_aligned[i] and volume_spike[i]
            short_entry = close[i] < l4_aligned[i] and volume_spike[i]
            
            # Exit when price returns to pivot level or volume drops
            long_exit = close[i] < pivot_aligned[i] or (not volume_spike[i])
            short_exit = close[i] > pivot_aligned[i] or (not volume_spike[i])
            
            if long_entry and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_entry and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and long_exit:
                position = 0
                signals[i] = 0.0
            elif position == -1 and short_exit:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # In ranging market (ADX <= 25), stay flat
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_vol_adx_v1"
timeframe = "1d"
leverage = 1.0