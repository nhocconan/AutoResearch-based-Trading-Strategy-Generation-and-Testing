#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ADX(14) as regime filter and 4h Donchian(20) breakout with volume confirmation.
# Long when 1d ADX < 25 (ranging market) AND price breaks above 4h Donchian upper band with volume spike (>1.8x median volume).
# Short when 1d ADX < 25 (ranging market) AND price breaks below 4h Donchian lower band with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. 1d ADX identifies ranging markets where mean reversion works best,
# Donchian breakout captures momentum in the direction of the range break, volume confirmation reduces false signals.
# Designed to work in both bull and bear markets by fading ranging conditions while following 4h momentum with precise entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) ===
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    up_move.iloc[0] = 0
    down_move.iloc[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = np.where(tr_14 != 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di = np.where(tr_14 != 0, 100 * minus_dm_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX condition: ranging market (ADX < 25)
    adx_ranging = adx < 25
    
    # Get 4h data for Donchian, volume, and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: Donchian(20), Volume Median, ATR(10) ===
    # Donchian Channels (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume median (20-period)
    vol_median_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    
    # ATR(10) for trailing stop
    tr1_4h = pd.Series(high_4h - low_4h)
    tr2_4h = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3_4h = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2_4h.iloc[0] = tr1_4h.iloc[0]
    tr3_4h.iloc[0] = tr1_4h.iloc[0]
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_10 = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (4h)
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20, 10)  # ADX(14), Donchian(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_ranging_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        adx_ranging = adx_ranging_aligned[i] > 0.5
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 4h volume for volume spike filter
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        current_vol_4h = vol_4h_aligned[i]
        
        # Volume spike filter: current 4h volume > 1.8x median volume
        volume_spike = current_vol_4h > (vol_median * 1.8)
        
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
            # 1d ADX ranging, price breaks above Donchian upper, volume spike
            if adx_ranging and price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # 1d ADX ranging, price breaks below Donchian lower, volume spike
            elif adx_ranging and price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dADX14_Ranging_4hDonchian20_Breakout_VolumeSpike1.8x_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0