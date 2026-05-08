#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily volume confirmation and ADX trend filter.
# Uses 4h price channel breakouts confirmed by daily volume > 1.5x 20-day average.
# In trending markets (ADX > 25), follows breakout direction.
# In ranging markets (ADX < 20), uses mean reversion at channel boundaries.
# Designed to work in both bull and bear markets by adapting to trend strength.
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Donchian_Breakout_Volume_ADX_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = np.full(len(df_daily), np.nan)
    vol_sum = 0
    for i in range(len(df_daily)):
        vol_sum += df_daily['volume'].iloc[i]
        if i >= 19:
            if i == 19:
                vol_avg_20[i] = vol_sum / 20
            else:
                vol_sum -= df_daily['volume'].iloc[i-20]
                vol_avg_20[i] = vol_sum / 20
    
    # Calculate daily ADX (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr = np.zeros(len(df_daily))
    for i in range(len(df_daily)):
        if i == 0:
            tr[i] = high_daily[i] - low_daily[i]
        else:
            tr[i] = max(
                high_daily[i] - low_daily[i],
                abs(high_daily[i] - close_daily[i-1]),
                abs(low_daily[i] - close_daily[i-1])
            )
    
    # Directional Movement
    dm_plus = np.zeros(len(df_daily))
    dm_minus = np.zeros(len(df_daily))
    for i in range(1, len(df_daily)):
        up_move = high_daily[i] - high_daily[i-1]
        down_move = low_daily[i-1] - low_daily[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smoothed values
    atr = np.zeros(len(df_daily))
    dm_plus_smooth = np.zeros(len(df_daily))
    dm_minus_smooth = np.zeros(len(df_daily))
    
    for i in range(len(df_daily)):
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
    di_plus = np.zeros(len(df_daily))
    di_minus = np.zeros(len(df_daily))
    dx = np.zeros(len(df_daily))
    for i in range(len(df_daily)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx = np.zeros(len(df_daily))
    for i in range(len(df_daily)):
        if i < 27:  # need 14+14 periods for smoothing
            adx[i] = np.nan
        elif i == 27:
            adx[i] = np.mean(dx[14:28])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align daily data to 4h timeframe
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current daily bar's data (last completed daily bar)
        idx_daily = 0
        while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
            idx_daily += 1
        idx_daily -= 1  # last completed daily bar
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_avg_20_current = vol_avg_20[idx_daily]
        adx_current = adx[idx_daily]
        
        if np.isnan(vol_avg_20_current) or np.isnan(adx_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        vol_current = df_daily['volume'].iloc[idx_daily]
        vol_confirmed = vol_current > 1.5 * vol_avg_20_current
        
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