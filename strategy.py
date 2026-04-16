#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian channel breakout (20) with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 1d ADX > 25 (strong trend) AND volume > 1.5x average.
# Short when price breaks below Donchian lower band AND 1d ADX > 25 AND volume > 1.5x average.
# Exit when price returns to Donchian middle band (mean reversion) or ADX < 20 (trend weakens).
# Uses discrete position size 0.25. Donchian captures breakouts, ADX filters ranging markets, volume confirms conviction.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (catch upside breakouts) and bear markets (catch downside breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series_12h = pd.Series(high_12h)
    low_series_12h = pd.Series(low_12h)
    donchian_upper_12h = high_series_12h.rolling(window=20, min_periods=20).max().values
    donchian_lower_12h = low_series_12h.rolling(window=20, min_periods=20).min().values
    donchian_middle_12h = (donchian_upper_12h + donchian_lower_12h) / 2.0
    
    # === 1d Indicators: ADX (14) for trend strength ===
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    close_series_1d = pd.Series(close_1d)
    
    # True Range
    tr1 = high_series_1d - low_series_1d
    tr2 = abs(high_series_1d - close_series_1d.shift(1))
    tr3 = abs(low_series_1d - close_series_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_series_1d - high_series_1d.shift(1)
    down_move = low_series_1d.shift(1) - low_series_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume average (20-period) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian20 + ADX14 need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol = volume[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = vol > 1.5 * vol_ma
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle band OR ADX < 20 (trend weakens)
            if (price <= middle) or (adx_val < 20):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle band OR ADX < 20 (trend weakens)
            if (price >= middle) or (adx_val < 20):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band AND ADX > 25 (strong uptrend) AND volume confirmed
            if (price > upper) and (adx_val > 25) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower band AND ADX > 25 (strong downtrend) AND volume confirmed
            elif (price < lower) and (adx_val > 25) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_1dADX_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0