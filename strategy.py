#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based volatility breakout with volume confirmation and ADX regime filter.
# Long when price breaks above 1d close + 0.5 * 1d ATR(14) with 4h volume spike (>2.0x median) and ADX>20.
# Short when price breaks below 1d close - 0.5 * 1d ATR(14) with same filters.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Target: 50-120 total trades over 4 years (12-30/year).
# ATR breakout captures volatility expansion, volume confirms conviction, ADX filters chop,
# ATR stop adapts to volatility and reduces whipsaws in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for ATR breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ATR(14) for breakout levels ===
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Breakout levels: close ± 0.5 * ATR
    upper_break_1d = close_1d + 0.5 * atr_14_1d
    lower_break_1d = close_1d - 0.5 * atr_14_1d
    
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
    tr1_4h = pd.Series(high_4h - low_4h)
    tr2_4h = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3_4h = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2_4h.iloc[0] = tr1_4h.iloc[0]
    tr3_4h.iloc[0] = tr1_4h.iloc[0]
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_temp = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values  # ATR for DX calculation
    
    # +DM and -DM
    up_move = pd.Series(high_4h).diff()
    down_move = pd.Series(low_4h).diff()
    up_move.iloc[0] = 0
    down_move.iloc[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    tr_period = 14
    atr_14 = pd.Series(tr_4h).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_14
    minus_di = 100 * minus_dm_smooth / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # ATR(10) for trailing stop
    atr_10 = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (4h)
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break_1d)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break_1d)
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
        if (np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        upper_break = upper_break_aligned[i]
        lower_break = lower_break_aligned[i]
        vol_median = vol_median_aligned[i]
        adx_val = adx_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 4h volume for volume spike filter
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        current_vol_4h = vol_4h_aligned[i]
        
        # Volume spike filter: current 4h volume > 2.0x median volume (stricter to reduce trades)
        volume_spike = current_vol_4h > (vol_median * 2.0)
        
        # Trend filter: ADX > 20 (lower threshold to capture more trends but still filter chop)
        trending = adx_val > 20
        
        # Breakout conditions
        breakout_long = price > upper_break
        breakout_short = price < lower_break
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr:
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
            # Breakout above upper level, volume spike, and trending market (ADX>20)
            if breakout_long and volume_spike and trending:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Breakout below lower level, volume spike, and trending market (ADX>20)
            elif breakout_short and volume_spike and trending:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_ATRBreakout_VolumeSpike2.0x_ADX20_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0