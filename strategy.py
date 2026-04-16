#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly ADX trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian upper band AND weekly ADX > 25 (strong trend) AND 1d volume > 1.5x 20-period average.
# Short when price breaks below 20-period Donchian lower band AND weekly ADX > 25 AND 1d volume > 1.5x 20-period average.
# Uses discrete position size 0.30. Donchian breakouts capture momentum, ADX filters for trending markets only, volume confirms participation.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets by requiring strong weekly trend.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag while capturing strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get weekly data once before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation (14+14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: ADX (14-period) ===
    # True Range
    tr1 = pd.Series(high_1w).diff().abs()
    tr2 = (pd.Series(high_1w) - pd.Series(close_1w).shift(1)).abs()
    tr3 = (pd.Series(low_1w) - pd.Series(close_1w).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = -pd.Series(low_1w).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1w
    minus_di = 100 * minus_dm_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20+14+14=48 periods needed)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        dc_upper = donchian_upper[i]
        dc_lower = donchian_lower[i]
        adx = adx_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band or ADX weakens (<20) or volume spike ends
            if price < dc_lower or adx < 20 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band or ADX weakens (<20) or volume spike ends
            if price > dc_upper or adx < 20 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND weekly ADX > 25 (strong trend) AND volume spike
            if price > dc_upper and adx > 25 and vol_spike:
                signals[i] = 0.30
                position = 1
            
            # SHORT: Price breaks below Donchian lower band AND weekly ADX > 25 (strong trend) AND volume spike
            elif price < dc_lower and adx > 25 and vol_spike:
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "1d_Donchian20_WeeklyADX25_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0