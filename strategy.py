#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout + volume spike + ADX trend filter
# Donchian breakout captures momentum, volume spike confirms institutional interest,
# ADX > 25 ensures trending market to avoid false breakouts in chop.
# Works in bull/bear: breakouts work in both directions, volume confirms validity,
# ADX filter prevents entries in low-volatility environments.
# Target: 20-50 trades/year, position size 0.25

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
    
    # Calculate 1d Donchian channels (20-period high/low)
    donch_len = 20
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower bands
    upper = pd.Series(high_1d).rolling(window=donch_len, min_periods=donch_len).max().values
    lower = pd.Series(low_1d).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Align Donchian to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Load 1d data ONCE for ADX (trend filter)
    high_1d_adx = df_1d['high'].values
    low_1d_adx = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # Calculate 1d ADX (14 periods)
    adx_len = 14
    tr1 = high_1d_adx[1:] - low_1d_adx[1:]
    tr2 = np.abs(high_1d_adx[1:] - close_1d_adx[:-1])
    tr3 = np.abs(low_1d_adx[1:] - close_1d_adx[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high_1d_adx[1:] - high_1d_adx[:-1]) > (low_1d_adx[:-1] - low_1d_adx[1:]), 
                       np.maximum(high_1d_adx[1:] - high_1d_adx[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d_adx[:-1] - low_1d_adx[1:]) > (high_1d_adx[1:] - high_1d_adx[:-1]), 
                        np.maximum(low_1d_adx[:-1] - low_1d_adx[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume average (20-period) for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, donch_len + adx_len + 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume spike: current volume > 1.5x 20-day average
        volume_spike = vol > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + trend + volume
            if price > upper_aligned[i] and trending and volume_spike:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below Donchian lower + trend + volume
            elif price < lower_aligned[i] and trending and volume_spike:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price retouches Donchian middle OR ADX drops below 20
            mid = (upper_aligned[i] + lower_aligned[i]) / 2
            if price < mid or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price retouches Donchian middle OR ADX drops below 20
            mid = (upper_aligned[i] + lower_aligned[i]) / 2
            if price > mid or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dDonchian_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0