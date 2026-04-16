#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Supertrend(10,3.0) for HTF trend direction + 6h Donchian(20) breakout with volume confirmation.
# Long when price > 1d Supertrend AND breaks above 6h Donchian upper(20) with volume spike (>1.8x median volume).
# Short when price < 1d Supertrend AND breaks below 6h Donchian lower(20) with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Volume spike and ATR trailing stop reduce whipsaw and overtrading.
# Supertrend adapts to volatility, making it effective in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Supertrend(10,3.0) ===
    # ATR(10)
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2_1d.iloc[0] = tr1_1d.iloc[0]
    tr3_1d.iloc[0] = tr1_1d.iloc[0]
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_10_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upperband and Lowerband
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr_10_1d
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr_10_1d
    
    # Final Upperband
    final_ub = np.zeros(len(close_1d))
    final_ub[0] = basic_ub[0]
    for i in range(1, len(close_1d)):
        if close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = max(basic_ub[i], final_ub[i-1])
        else:
            final_ub[i] = basic_ub[i]
    
    # Final Lowerband
    final_lb = np.zeros(len(close_1d))
    final_lb[0] = basic_lb[0]
    for i in range(1, len(close_1d)):
        if close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = min(basic_lb[i], final_lb[i-1])
        else:
            final_lb[i] = basic_lb[i]
    
    # Supertrend
    supertrend_1d = np.zeros(len(close_1d))
    supertrend_1d[0] = final_ub[0]
    for i in range(1, len(close_1d)):
        if supertrend_1d[i-1] == final_ub[i-1]:
            if close_1d[i] <= final_ub[i]:
                supertrend_1d[i] = final_ub[i]
            else:
                supertrend_1d[i] = final_lb[i]
        else:
            if close_1d[i] >= final_lb[i]:
                supertrend_1d[i] = final_lb[i]
            else:
                supertrend_1d[i] = final_ub[i]
    
    # Get 6h data for Donchian, volume, and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 6h Indicators: Donchian(20), Volume Median, ATR(10) ===
    # Donchian channels (20-period)
    donchian_upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume median (20-period)
    vol_median_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).median().values
    
    # ATR(10) for trailing stop
    tr1_6h = pd.Series(high_6h - low_6h)
    tr2_6h = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3_6h = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr2_6h.iloc[0] = tr1_6h.iloc[0]
    tr3_6h.iloc[0] = tr1_6h.iloc[0]
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_10 = pd.Series(tr_6h).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (6h)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(10, 20, 10)  # Supertrend ATR(10), Donchian(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(atr_aligned[i])):
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
        price = close[i]
        
        # Get current 6h volume for volume spike filter
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        current_vol_6h = vol_6h_aligned[i]
        
        # Volume spike filter: current 6h volume > 1.8x median volume
        volume_spike = current_vol_6h > (vol_median * 1.8)
        
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
            # Price above 1d Supertrend (uptrend), breakout above Donchian upper, volume spike
            if price > supertrend_val and price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Price below 1d Supertrend (downtrend), breakout below Donchian lower, volume spike
            elif price < supertrend_val and price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dSupertrend10_3_6hDonchian20_Breakout_VolumeSpike1.8x_ATRTrail2.5_v1"
timeframe = "6h"
leverage = 1.0