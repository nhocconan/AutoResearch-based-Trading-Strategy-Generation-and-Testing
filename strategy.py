#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike confirmation and 1d ADX(14) trend filter.
# Long when price > upper band, 1d volume > 2.0x its 20-period median, and daily ADX > 25 (trending market).
# Short when price < lower band, same volume spike condition, and daily ADX > 25.
# Exit when price crosses middle band (mean reversion).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Combines price channel breakout with volume spike filter and trend filter for robustness in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian channels (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian channels
    donchian_upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # Get daily data for volume spike filter and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Indicators: Volume spike filter ===
    vol_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # === Daily Indicators: ADX(14) trend filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Align daily volume for volume confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 30, 14)  # 12h Donchian, daily volume median, daily ADX
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_median = vol_median_aligned[i]
        adx = adx_14_aligned[i]
        daily_volume = vol_1d_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Donchian middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Donchian middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current daily volume > 2.0x its 20-period median
            volume_spike = daily_volume > (vol_median * 2.0)
            # Trend filter: ADX > 25 indicates trending market
            trending_market = adx > 25.0
            
            # LONG CONDITIONS
            # Price breaks above Donchian upper band AND volume spike AND trending market
            if price > upper and volume_spike and trending_market:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band AND volume spike AND trending market
            elif price < lower and volume_spike and trending_market:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_1dVolumeSpike2.0x_1dADX25_v1"
timeframe = "12h"
leverage = 1.0