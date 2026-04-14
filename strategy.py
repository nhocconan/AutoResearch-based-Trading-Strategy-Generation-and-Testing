#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation and ADX Trend Filter
# Donchian Channel breakouts capture volatility expansion and trend continuation
# Volume > 1.5x average confirms breakout strength
# Weekly ADX > 25 ensures we only trade in trending markets (works in bull and bear)
# Exit when price returns to middle of Donchian channel (mean reversion within trend)
# Target: 15-25 trades/year per symbol to avoid fee drag on daily timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for ADX and Donchian
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX (14 periods)
    adx_len = 14
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_weekly = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx_weekly)
    
    # Weekly Donchian Channel (20 periods)
    donch_len = 20
    upper_donch = pd.Series(high_w).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_donch = pd.Series(low_w).rolling(window=donch_len, min_periods=donch_len).min().values
    mid_donch = (upper_donch + lower_donch) / 2
    
    # Align Donchian bands to daily timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_weekly, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_weekly, lower_donch)
    mid_donch_aligned = align_htf_to_ltf(prices, df_weekly, mid_donch)
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, donch_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_donch_aligned[i]) or
            np.isnan(lower_donch_aligned[i]) or
            np.isnan(mid_donch_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above weekly upper Donchian + volume + trend
            if (close[i] > upper_donch_aligned[i-1] and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below weekly lower Donchian + volume + trend
            elif (close[i] < lower_donch_aligned[i-1] and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly middle Donchian (mean reversion)
            if close[i] < mid_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly middle Donchian (mean reversion)
            if close[i] > mid_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX_v1"
timeframe = "1d"
leverage = 1.0