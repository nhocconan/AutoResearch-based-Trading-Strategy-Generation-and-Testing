# 1d_RangeBreakout_VolumeTrend_Strategy
# Hypothesis: Daily breakouts of weekly Donchian channels with volume confirmation and weekly ADX filter
# capture institutional momentum while avoiding whipsaws. Weekly timeframe filters for strong trends,
# daily provides timely entries. Volume confirms institutional participation. Designed for 20-35 trades/year.

name = "1d_RangeBreakout_VolumeTrend_Strategy"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for calculations
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    donchian_high = np.full_like(high_weekly, np.nan)
    donchian_low = np.full_like(high_weekly, np.nan)
    
    for i in range(len(high_weekly)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(high_weekly[i-19:i+1])
            donchian_low[i] = np.min(low_weekly[i-19:i+1])
    
    # Align Donchian levels to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Calculate weekly ADX (14-period) for trend strength
    # True Range
    tr1 = high_weekly[1:] - low_weekly[1:]
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1]) if 'close_weekly' in locals() else np.abs(high_weekly[1:] - df_weekly['close'].values[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1]) if 'close_weekly' in locals() else np.abs(low_weekly[1:] - df_weekly['close'].values[:-1])
    
    close_weekly = df_weekly['close'].values
    tr1 = high_weekly[1:] - low_weekly[1:]
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_weekly[1:] - high_weekly[:-1]) > (low_weekly[:-1] - low_weekly[1:]), 
                       np.maximum(high_weekly[1:] - high_weekly[:-1], 0), 0)
    dm_minus = np.where((low_weekly[:-1] - low_weekly[1:]) > (high_weekly[1:] - high_weekly[:-1]), 
                        np.maximum(low_weekly[:-1] - low_weekly[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    tr_sum = np.full_like(high_weekly, np.nan)
    dm_plus_sum = np.full_like(high_weekly, np.nan)
    dm_minus_sum = np.full_like(high_weekly, np.nan)
    
    for i in range(len(high_weekly)):
        if i >= 13:  # 14-period smoothing
            tr_sum[i] = np.nansum(tr[i-13:i+1])
            dm_plus_sum[i] = np.nansum(dm_plus[i-13:i+1])
            dm_minus_sum[i] = np.nansum(dm_minus[i-13:i+1])
    
    # Directional Indicators
    di_plus = np.full_like(high_weekly, np.nan)
    di_minus = np.full_like(high_weekly, np.nan)
    dx = np.full_like(high_weekly, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    di_plus[valid] = 100 * dm_plus_sum[valid] / tr_sum[valid]
    di_minus[valid] = 100 * dm_minus_sum[valid] / tr_sum[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(high_weekly, np.nan)
    for i in range(len(high_weekly)):
        if i >= 27:  # 14 + 13 for ADX smoothing
            valid_dx = dx[i-13:i+1]
            if not np.all(np.isnan(valid_dx)):
                adx[i] = np.nanmean(valid_dx)
    
    # Align ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 20)  # Ensure ADX and Donchian are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + ADX > 25 + volume confirmation
            if close[i] > donchian_high_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + ADX > 25 + volume confirmation
            elif close[i] < donchian_low_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low or ADX weakens
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high or ADX weakens
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals