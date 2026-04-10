#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# - Long when price breaks above 4h Donchian upper band AND 1d volume > 1.5x 20-period volume SMA AND 1w ADX > 25
# - Short when price breaks below 4h Donchian lower band AND 1d volume > 1.5x 20-period volume SMA AND 1w ADX > 25
# - Exit: price retreats to midpoint of Donchian channel OR volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 20-50 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian structure from 4h, volume confirmation from 1d, trend filter from 1w

name = "4h_1d_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w ADX for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder's smoothing
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX (smoothed DX)
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period]) if 2*period-1 < len(dx) else 0
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Align HTF data to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)  # 4h data, no alignment needed
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h volume SMA for confirmation
    volume_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_sma_20_4h[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_4h = volume[i] > 1.5 * volume_sma_20_4h[i]
        vol_confirm_1d = volume_1d[i // 6] > 1.5 * volume_sma_20_1d_aligned[i] if i // 6 < len(volume_1d) else False
        vol_confirm = vol_confirm_4h and vol_confirm_1d
        
        # Trend filter: 1w ADX > 25 (trending market)
        trend_strong = adx_aligned[i] > 25
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper_aligned[i-1]  # Break above previous upper band
        breakout_down = close[i] < donchian_lower_aligned[i-1]  # Break below previous lower band
        
        # Exit conditions: price retreats to midpoint OR loss of volume confirmation
        exit_long = close[i] < donchian_mid_aligned[i] or not vol_confirm
        exit_short = close[i] > donchian_mid_aligned[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_strong and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_strong and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals