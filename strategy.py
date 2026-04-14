#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1-day Donchian breakout with 1-week ADX trend filter.
# Long when price breaks above 20-period 1-day Donchian high with 1-week ADX > 25 (trending).
# Short when price breaks below 20-period 1-day Donchian low with 1-week ADX > 25 (trending).
# Uses volume confirmation (1.5x 20-period average) to filter false breakouts.
# Exit when price returns to opposite Donchian band or ADX falls below 20 (range).
# Designed for 12h timeframe to capture multi-day trends while avoiding false breakouts in ranging markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data ONCE for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initialize first values
    tr_smooth[atr_period] = np.nansum(tr[1:atr_period+1])
    dm_plus_smooth[atr_period] = np.nansum(dm_plus[1:atr_period+1])
    dm_minus_smooth[atr_period] = np.nansum(dm_minus[1:atr_period+1])
    
    # Wilder smoothing
    for i in range(atr_period + 1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / atr_period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / atr_period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / atr_period) + dm_minus[i]
    
    # Avoid division by zero
    tr_smooth = np.where(tr_smooth == 0, 1e-10, tr_smooth)
    
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[2*atr_period] = np.nanmean(dx[atr_period+1:2*atr_period+1])
    for i in range(2*atr_period + 1, len(dx)):
        adx[i] = (adx[i-1] * (atr_period - 1) + dx[i]) / atr_period
    
    # Align indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(40, 2*14+1)  # Need Donchian and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 for trending market
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # Exit when ranging
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high AND trending
            if (close[i] > donch_high_aligned[i] and 
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND trending
            elif (close[i] < donch_low_aligned[i] and 
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or ADX < 20 (ranging)
            if (close[i] <= donch_low_aligned[i] or 
                ranging):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian high or ADX < 20 (ranging)
            if (close[i] >= donch_high_aligned[i] or 
                ranging):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dDonchian_1wADX_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0