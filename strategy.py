#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w ADX filter and volume confirmation.
# Long when price breaks above 1d Donchian high(20) with 1w ADX > 20 and volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian low(20) with 1w ADX > 20 and volume > 1.5x 20-period average.
# Exit when price returns to 1d Donchian midpoint (mean reversion).
# Uses discrete position size 0.25. 1d Donchian provides structure from higher timeframe, 12h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: Donchian Channel (20) based on prior day ===
    # Calculate using prior day's high, low, close (shift by 1 to use completed day only)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(phigh).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(plow).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align daily Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Get weekly data once before loop for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: ADX (14) for trend strength filter ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing)
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Directional Index (DX) and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume moving average (20-period) on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        dch = donch_high_aligned[i]
        dcl = donch_low_aligned[i]
        dcm = donch_mid_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to daily Donchian midpoint
            if price <= dcm:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to daily Donchian midpoint
            if price >= dcm:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: only trade when ADX > 20 (moderate trend)
            trend_filter = adx_val > 20
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Price breaks above daily Donchian high with trend and volume confirmation
            if (price > dch) and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below daily Donchian low with trend and volume confirmation
            elif (price < dcl) and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dDonchian20_1wADX_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0