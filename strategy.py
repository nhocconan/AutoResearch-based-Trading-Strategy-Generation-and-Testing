#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily volume confirmation and ADX trend filter.
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-period average AND ADX > 25 (trending market).
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-period average AND ADX > 25.
# Exit when price crosses back inside the Donchian channel.
# Uses 4h timeframe with 1d volume for higher timeframe context.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled frequency to avoid fee drag.

name = "4h_Donchian_20_DailyVolume_ADX"
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
    
    # Daily data for volume confirmation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Daily volume filter: current volume > 1.5x 20-period average
    vol_ma20_d = pd.Series(df_d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma20_d_aligned = align_htf_to_ltf(prices, df_d, vol_ma20_d)
    volume_filter = volume > (1.5 * vol_ma20_d_aligned)
    
    # Donchian channels (20-period) on 4h data
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # ADX trend filter (14-period) on 4h data
    adx_period = 14
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(np.absolute(low - np.roll(close, 1)), tr1)
    tr = np.where(np.arange(len(close)) == 0, high - low, tr2)
    atr = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm = np.where(np.arange(len(close)) == 0, 0, plus_dm)
    minus_dm = np.where(np.arange(len(close)) == 0, 0, minus_dm)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values / atr
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Trending market condition: ADX > 25
    trending_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, adx_period)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trending_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band, volume filter, trending market
            long_cond = (close[i] > upper_band[i]) and volume_filter[i] and trending_filter[i]
            # Short conditions: price breaks below Donchian lower band, volume filter, trending market
            short_cond = (close[i] < lower_band[i]) and volume_filter[i] and trending_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower band
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper band
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals