#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly volume confirmation and ADX trend filter.
# Uses daily price channel breakouts confirmed by weekly volume > 1.8x 10-week average.
# In trending markets (weekly ADX > 25), follows breakout direction.
# In ranging markets (weekly ADX < 20), uses mean reversion at channel boundaries.
# Designed to work in both bull and bear markets by adapting to trend strength.
# Target: 10-25 trades/year (40-100 total over 4 years).

name = "1d_Donchian_Breakout_WeeklyVolume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for volume and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly volume average (10-period)
    vol_avg_10 = np.full(len(df_weekly), np.nan)
    vol_sum = 0
    for i in range(len(df_weekly)):
        vol_sum += df_weekly['volume'].iloc[i]
        if i >= 9:
            if i == 9:
                vol_avg_10[i] = vol_sum / 10
            else:
                vol_sum -= df_weekly['volume'].iloc[i-10]
                vol_avg_10[i] = vol_sum / 10
    
    # Calculate weekly ADX (14-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # True Range
    tr = np.zeros(len(df_weekly))
    for i in range(len(df_weekly)):
        if i == 0:
            tr[i] = high_weekly[i] - low_weekly[i]
        else:
            tr[i] = max(
                high_weekly[i] - low_weekly[i],
                abs(high_weekly[i] - close_weekly[i-1]),
                abs(low_weekly[i] - close_weekly[i-1])
            )
    
    # Directional Movement
    dm_plus = np.zeros(len(df_weekly))
    dm_minus = np.zeros(len(df_weekly))
    for i in range(1, len(df_weekly)):
        up_move = high_weekly[i] - high_weekly[i-1]
        down_move = low_weekly[i-1] - low_weekly[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smoothed values
    atr = np.zeros(len(df_weekly))
    dm_plus_smooth = np.zeros(len(df_weekly))
    dm_minus_smooth = np.zeros(len(df_weekly))
    
    for i in range(len(df_weekly)):
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
    di_plus = np.zeros(len(df_weekly))
    di_minus = np.zeros(len(df_weekly))
    dx = np.zeros(len(df_weekly))
    for i in range(len(df_weekly)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx = np.zeros(len(df_weekly))
    for i in range(len(df_weekly)):
        if i < 27:  # need 14+14 periods for smoothing
            adx[i] = np.nan
        elif i == 27:
            adx[i] = np.mean(dx[14:28])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align weekly data to daily timeframe
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_weekly, vol_avg_10)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_10_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current weekly bar's data (last completed weekly bar)
        idx_weekly = 0
        while idx_weekly < len(df_weekly) and df_weekly.iloc[idx_weekly]['open_time'] <= prices.iloc[i]['open_time']:
            idx_weekly += 1
        idx_weekly -= 1  # last completed weekly bar
        
        if idx_weekly < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_avg_10_current = vol_avg_10[idx_weekly]
        adx_current = adx[idx_weekly]
        
        if np.isnan(vol_avg_10_current) or np.isnan(adx_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current weekly volume > 1.8x 10-week average
        vol_current = df_weekly['volume'].iloc[idx_weekly]
        vol_confirmed = vol_current > 1.8 * vol_avg_10_current
        
        # Trend detection
        is_trending = adx_current > 25
        is_ranging = adx_current < 20
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                if is_trending:
                    # In trending market: breakout continuation
                    if close[i] > donchian_high[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < donchian_low[i]:
                        signals[i] = -0.25
                        position = -1
                elif is_ranging:
                    # In ranging market: mean reversion at channel boundaries
                    if close[i] <= donchian_low[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] >= donchian_high[i]:
                        signals[i] = -0.25
                        position = -1
                else:
                    # Transition zone: wait for clearer signal
                    pass
        elif position == 1:
            # Manage long position
            exit_signal = False
            if is_trending and close[i] < donchian_low[i]:
                exit_signal = True  # breakout failed
            elif is_ranging and close[i] >= donchian_high[i]:
                exit_signal = True  # mean reversion target reached
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
            if is_trending and close[i] > donchian_high[i]:
                exit_signal = True  # breakout failed
            elif is_ranging and close[i] <= donchian_low[i]:
                exit_signal = True  # mean reversion target reached
            elif not vol_confirmed:
                exit_signal = True  # volume confirmation lost
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals