#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h volume spike (2x median) and 1d ADX regime filter (>25)
# Long when price > Camarilla R1 AND 12h volume > 2x 20-period median AND 1d ADX > 25
# Short when price < Camarilla S1 AND 12h volume > 2x 20-period median AND 1d ADX > 25
# Exit when price crosses Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete position size 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
# Combines pivot level breakout with volume confirmation and trend regime filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_1d + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Volume median (20-period) ===
    volume_12h = df_12h['volume'].values
    vol_median_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    vol_median_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20_12h)
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Note: Using 4h OHLC to approximate daily levels for intraday precision
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_4h + low_4h + close_4h) / 3.0
    
    # Camarilla levels (based on previous 4h bar's range)
    range_4h = high_4h - low_4h
    camarilla_pivot = typical_price
    camarilla_r1 = camarilla_pivot + (range_4h * 1.1 / 12)
    camarilla_s1 = camarilla_pivot - (range_4h * 1.1 / 12)
    
    # Align Camarilla levels to primary timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14, 20)  # 1d ADX, 4h Camarilla, 12h volume
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_median_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 2x 20-period 12h volume median
        vol_threshold = vol_median_20_12h_aligned[i] * 2.0
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Regime filter: 1d ADX > 25 (trending market)
        regime_filter = adx_1d_aligned[i] > 25.0
        
        # Price levels
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Camarilla pivot (mean reversion)
            if price < pivot:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Camarilla pivot (mean reversion)
            if price > pivot:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Camarilla R1 AND volume confirmation AND trending regime
            if price > r1 and vol_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S1 AND volume confirmation AND trending regime
            elif price < s1 and vol_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Camarilla_R1S1_12hVolume2x_1dADX25_v1"
timeframe = "4h"
leverage = 1.0