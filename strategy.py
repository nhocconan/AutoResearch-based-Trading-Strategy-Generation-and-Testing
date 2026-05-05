#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and ADX trend filter
# Long when price breaks above weekly Donchian(20) high AND volume > 2.0 * 20-day avg volume AND ADX(14) > 25
# Short when price breaks below weekly Donchian(20) low AND volume > 2.0 * 20-day avg volume AND ADX(14) > 25
# Exit when price crosses weekly Donchian(20) midpoint OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Donchian provides structural support/resistance levels that work in both bull and bear markets
# Volume confirmation ensures breakout strength and reduces false signals
# ADX filter ensures we only trade in trending markets, avoiding chop

name = "1d_Donchian20_1wEMA50_VolumeSpike_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channel (20-period)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high_20
    donchian_low = lowest_low_20
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly Donchian levels to daily timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get daily data ONCE before loop for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average and ADX
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume for confirmation
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on daily timeframe
    # ADX calculation requires +DI, -DI, and DX
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - close_1d.shift(1)))
    tr3 = pd.Series(abs(low_1d - close_1d.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    up_move = high_1d - high_1d.shift(1)
    down_move = low_1d.shift(1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily indicators to daily timeframe (no additional delay needed for these)
    # Since we're using daily data for daily timeframe, alignment is direct
    # But we still use the helper for consistency and proper handling
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high, volume confirmation, strong trend (ADX > 25), in session
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > (2.0 * avg_volume_20_aligned[i]) and 
                adx_aligned[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian low, volume confirmation, strong trend (ADX > 25), in session
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > (2.0 * avg_volume_20_aligned[i]) and 
                  adx_aligned[i] > 25):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or volume[i] < avg_volume_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or volume[i] < avg_volume_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals