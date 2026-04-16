#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1, S1) for breakout entries,
# confirmed by 4h volume spike (>1.5x 20-period median) and 4h ADX (>25) for trending regime.
# Long when price breaks above R1 with volume spike and ADX>25.
# Short when price breaks below S1 with volume spike and ADX>25.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 3.0*ATR,
# short exits when price > lowest low since entry + 3.0*ATR.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla pivots provide intraday support/resistance, volume confirms conviction,
# ADX filters for trending markets to avoid chop, ATR stop reduces whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (high + low + close) / 3
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Get 4h data for volume, ADX, and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: Volume Median, ADX(14), ATR(10) ===
    # Volume median (20-period)
    vol_median_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    
    # ADX(14)
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_temp = tr.rolling(window=14, min_periods=14).mean().values  # ATR for DX calculation
    
    # +DM and -DM
    up_move = pd.Series(high_4h).diff()
    down_move = pd.Series(low_4h).diff()
    up_move.iloc[0] = 0
    down_move.iloc[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    tr_period = 14
    atr_14 = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_14
    minus_di = 100 * minus_dm_smooth / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # ATR(10) for trailing stop
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (4h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 10)  # volume median(20), ADX(14), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_median = vol_median_aligned[i]
        adx_val = adx_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 4h volume for volume spike filter
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        current_vol_4h = vol_4h_aligned[i]
        
        # Volume spike filter: current 4h volume > 1.5x median volume
        volume_spike = current_vol_4h > (vol_median * 1.5)
        
        # Trend filter: ADX > 25
        trending = adx_val > 25
        
        # Breakout conditions
        breakout_long = price > r1
        breakout_short = price < s1
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 3.0*ATR
            if price < highest_since_entry - 3.0 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 3.0*ATR
            if price > lowest_since_entry + 3.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Breakout above R1, volume spike, and trending market (ADX>25)
            if breakout_long and volume_spike and trending:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Breakout below S1, volume spike, and trending market (ADX>25)
            elif breakout_short and volume_spike and trending:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_CamarillaR1S1_Breakout_VolumeSpike1.5x_ADX25_ATRTrail3.0_v1"
timeframe = "4h"
leverage = 1.0