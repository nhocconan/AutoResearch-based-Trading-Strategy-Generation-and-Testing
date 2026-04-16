#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Supertrend(ATR=10, mult=3) for trend + 4h Donchian(20) breakout with volume confirmation (>1.5x median) and chop regime filter (chop < 61.8).
# Long when price > 1d Supertrend AND breaks above 4h Donchian upper(20) with volume spike AND chop < 61.8.
# Short when price < 1d Supertrend AND breaks below 4h Donchian lower(20) with volume spike AND chop < 61.8.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.0*ATR, short exits when price > lowest low since entry + 2.0*ATR.
# Uses discrete position size 0.25. Supertrend adapts to volatility, making it robust in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Supertrend(10, 3) ===
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2
    basic_ub = hl2 + (multiplier * atr_1d)
    basic_lb = hl2 - (multiplier * atr_1d)
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_1d))
    final_lb = np.zeros(len(close_1d))
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close_1d)):
        if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(len(close_1d))
    supertrend[0] = final_ub[0]
    for i in range(1, len(close_1d)):
        if supertrend[i-1] == final_ub[i-1]:
            if close_1d[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if close_1d[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Get 4h data for Donchian, volume, ATR, and chop filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: Donchian(20), Volume Median, ATR(10), Chop Filter ===
    # Donchian channels (20-period)
    donchian_upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
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
    
    # Choppiness Index (14) for regime filter
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    # Avoid division by zero
    chop = 100 * (np.log10(sum_atr_14) - np.log10(range_14)) / np.log10(14)
    chop = np.where((range_14 == 0) | np.isnan(chop), 50.0, chop)
    
    # Align all indicators to primary timeframe (4h)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14, 10)  # Supertrend(30), Donchian(20), Chop(14), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        supertrend_val = supertrend_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        chop_val = chop_aligned[i]
        price = close[i]
        
        # Get current 4h volume for volume spike filter
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        current_vol_4h = vol_4h_aligned[i]
        
        # Volume spike filter: current 4h volume > 1.5x median volume
        volume_spike = current_vol_4h > (vol_median * 1.5)
        
        # Regime filter: chop < 61.8 indicates trending market (not choppy/ranging)
        trending = chop_val < 61.8
        
        # Breakout conditions
        breakout_long = price > upper
        breakout_short = price < lower
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 2.0*ATR
            if price < highest_since_entry - 2.0 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 2.0*ATR
            if price > lowest_since_entry + 2.0 * atr:
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
            # Price above 1d Supertrend (uptrend), breakout above Donchian upper, volume spike, trending market
            if price > supertrend_val and breakout_long and volume_spike and trending:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Price below 1d Supertrend (downtrend), breakout below Donchian lower, volume spike, trending market
            elif price < supertrend_val and breakout_short and volume_spike and trending:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dSupertrend10_3_4hDonchian20_Breakout_VolumeSpike1.5x_Chop61.8_ATRTrail2.0_v1"
timeframe = "4h"
leverage = 1.0