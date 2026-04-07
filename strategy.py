#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + Volume + ADX Trend Filter
# Hypothesis: Breakouts of 12h Donchian(20) channels with volume confirmation
# and trend alignment from daily ADX. Works in bull/bear by following trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_donchian20_volume_adx_v14"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Upper and lower bands from previous day
    upper = pd.Series(high_daily).rolling(window=20, min_periods=20).max().shift(1).values
    lower = pd.Series(low_daily).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h
    upper_12h = align_htf_to_ltf(prices, df_daily, upper)
    lower_12h = align_htf_to_ltf(prices, df_daily, lower)
    
    # Daily ADX for trend strength (14-period)
    # Calculate True Range
    high_low = high_daily[1:] - low_daily[1:]
    high_close = np.abs(high_daily[1:] - close_daily[:-1])
    low_close = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr = np.concatenate([[np.nan], tr])  # align with daily index
    
    # Directional Movement
    up_move = high_daily[1:] - high_daily[:-1]
    down_move = low_daily[:-1] - low_daily[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h
    adx_12h = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume filter: 12h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Trend filter: ADX > 25 for trending market
        trend_ok = adx_12h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend weakens
            if low[i] < lower_12h[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend weakens
            if high[i] > upper_12h[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of trend with volume and ADX confirmation
            if vol_ok and trend_ok:
                if high[i] > upper_12h[i]:  # Breakout above upper band
                    position = 1
                    signals[i] = 0.25
                elif low[i] < lower_12h[i]:  # Breakdown below lower band
                    position = -1
                    signals[i] = -0.25
    
    return signals