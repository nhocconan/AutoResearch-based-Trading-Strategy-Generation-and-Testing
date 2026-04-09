#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian breakout with 1d ADX trend filter and volume confirmation
# - Uses 12h HTF for Donchian channel (20-period) to identify breakouts
# - Uses 1d HTF for ADX (14-period) to filter only strong trends (ADX > 25)
# - Volume confirmation: current 6h volume > 1.5x 20-period average to avoid low-quality breakouts
# - Long when price breaks above 12h Donchian upper band in strong uptrend (ADX > 25)
# - Short when price breaks below 12h Donchian lower band in strong downtrend (ADX > 25)
# - Exit when price returns to the middle of the Donchian channel or trend weakens (ADX < 20)
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_1d_donchian_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 12h Donchian channel (20 periods)
    period20_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d ADX (14 periods)
    # True Range
    tr1 = pd.Series(high_1d).rolling(window=2).max().values - pd.Series(low_1d).rolling(window=2).min().values
    tr2 = np.abs(pd.Series(high_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).rolling(window=2).shift(1).values - pd.Series(close_1d).rolling(window=2).shift(1).values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff().values
    down_move = -pd.Series(low_1d).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # ADX trend strength filters
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: price returns to middle or trend weakens
            if close[i] <= donchian_middle_aligned[i] or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: price returns to middle or trend weakens
            if close[i] >= donchian_middle_aligned[i] or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: Donchian breakout with strong trend and volume confirmation
            if volume_confirmed and strong_trend:
                if close[i] > donchian_upper_aligned[i]:
                    # Break above upper band: long
                    position = 1
                    signals[i] = position_size
                elif close[i] < donchian_lower_aligned[i]:
                    # Break below lower band: short
                    position = -1
                    signals[i] = -position_size
    
    return signals