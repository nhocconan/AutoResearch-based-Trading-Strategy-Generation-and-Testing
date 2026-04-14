#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and ADX trend filter.
# Long when price breaks above 12h Donchian upper band with ADX > 25 and volume > 1.5x average.
# Short when price breaks below 12h Donchian lower band with ADX > 25 and volume > 1.5x average.
# Exit when price returns to 12h Donchian middle band or ADX < 20.
# Designed to capture strong trends while filtering choppy markets.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Donchian channels and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period)
    donch_len = 20
    upper_12h = pd.Series(high_12h).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_12h = pd.Series(low_12h).rolling(window=donch_len, min_periods=donch_len).min().values
    middle_12h = (upper_12h + lower_12h) / 2
    
    # Calculate ADX (14-period) on 12h
    adx_period = 14
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    up_move = np.concatenate([[np.nan], up_move])
    down_move = np.concatenate([[np.nan], down_move])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Align indicators to lower timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_12h_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donch_len, adx_period) + 5
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or
            np.isnan(middle_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 for strong trend
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above upper band AND strong trend
            if (close[i] > upper_12h_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band AND strong trend
            elif (close[i] < lower_12h_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or trend weakens
            if (close[i] < middle_12h_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band or trend weakens
            if (close[i] > middle_12h_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hDonchian_Breakout_ADX_Volume_v1"
timeframe = "4h"
leverage = 1.0