#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX regime filter
# Donchian breakouts capture strong momentum moves; volume confirms institutional participation
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Works in both bull and bear markets by trading breakouts in direction of the trend
# Target: 20-40 trades/year (80-160 total over 4 years)

name = "4h_Donchian20_1dVolume_ADX_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate daily volume MA for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate daily ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(55, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume_confirm = volume_confirm[i]
        curr_adx = adx_aligned[i]
        curr_vol_ma_1d = vol_ma_20_1d_aligned[i]
        curr_vol_1d = vol_1d[-1] if len(vol_1d) > 0 else 0  # simplified for alignment
        
        # Volume spike: current daily volume > 1.8x 20-day MA
        # Simplified: use 4h volume proxy for daily confirmation
        volume_spike = curr_volume_confirm and (volume[i] > 1.8 * vol_ma_20[i])
        
        # Trend filter: ADX > 25 indicates trending market
        trending = curr_adx > 25
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in trending market
            if volume_spike and trending:
                # Bullish breakout: price breaks above upper Donchian channel
                if curr_close > curr_highest_20:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian channel
                elif curr_close < curr_lowest_20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to midpoint of Donchian channel OR breaks below lower channel with volume
            midpoint = (curr_highest_20 + curr_lowest_20) / 2.0
            
            if curr_close <= midpoint or (curr_close < curr_lowest_20 and volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to midpoint of Donchian channel OR breaks above upper channel with volume
            midpoint = (curr_highest_20 + curr_lowest_20) / 2.0
            
            if curr_close >= midpoint or (curr_close > curr_highest_20 and volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals