#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_DonchianBreakout_1dVolume_ADX_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr[i] = high_1d[i] - low_1d[i]
        else:
            tr[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # Directional Movement
    dm_plus = np.zeros(len(df_1d))
    dm_minus = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smoothed values
    atr = np.zeros(len(df_1d))
    dm_plus_smooth = np.zeros(len(df_1d))
    dm_minus_smooth = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 14:
            if i == 0:
                atr[i] = tr[i]
                dm_plus_smooth[i] = dm_plus[i]
                dm_minus_smooth[i] = dm_minus[i]
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.zeros(len(df_1d))
    di_minus = np.zeros(len(df_1d))
    dx = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 27:  # need 14+14 periods for smoothing
            adx[i] = np.nan
        elif i == 27:
            adx[i] = np.mean(dx[14:28])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d data to 12h timeframe
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 1d bar's data (last completed 1d bar)
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed 1d bar
        
        if idx_1d < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_avg_20_current = vol_avg_20[idx_1d]
        adx_current = adx[idx_1d]
        
        if np.isnan(vol_avg_20_current) or np.isnan(adx_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_current = df_1d['volume'].iloc[idx_1d]
        vol_confirmed = vol_current > 1.5 * vol_avg_20_current
        
        # Trend detection
        is_trending = adx_current > 25
        is_ranging = adx_current < 20
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                if is_trending:
                    # In trending market: Donchian breakout
                    if close[i] > donchian_high[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < donchian_low[i]:
                        signals[i] = -0.25
                        position = -1
                elif is_ranging:
                    # In ranging market: mean reversion at Donchian mid-point
                    if close[i] < donchian_low[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] > donchian_high[i]:
                        signals[i] = -0.25
                        position = -1
                else:
                    # Transition zone: wait for clearer signal
                    pass
        elif position == 1:
            # Manage long position
            exit_signal = False
            if is_trending:
                # Exit when price breaks below Donchian low
                if close[i] < donchian_low[i]:
                    exit_signal = True
            elif is_ranging:
                # Exit when price reaches Donchian mid-point
                if close[i] >= donchian_mid[i]:
                    exit_signal = True
            elif not vol_confirmed:
                exit_signal = True  # volume confirmation lost
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            if is_trending:
                # Exit when price breaks above Donchian high
                if close[i] > donchian_high[i]:
                    exit_signal = True
            elif is_ranging:
                # Exit when price reaches Donchian mid-point
                if close[i] <= donchian_mid[i]:
                    exit_signal = True
            elif not vol_confirmed:
                exit_signal = True  # volume confirmation lost
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals