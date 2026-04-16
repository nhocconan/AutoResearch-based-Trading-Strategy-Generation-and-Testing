#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Bollinger Band breakout with volume confirmation and chop regime filter.
# Long when price breaks above upper BB(20,2.0) with volume spike (>1.8x median) and chop < 61.8 (trending).
# Short when price breaks below lower BB(20,2.0) with same filters.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Tight volume and chop filters reduce overtrading.
# Bollinger Bands adapt to volatility, working in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Bollinger Bands (20,2.0) ===
    # Middle band: 20-period SMA
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    # Get 4h data for volume, ATR, and chop filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: Volume Median, ATR(10), Chop Filter ===
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
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    # Avoid division by zero
    chop = 100 * (np.log10(sum_atr_14) - np.log10(range_14)) / np.log10(14)
    # Handle invalid values (set to 50 as neutral)
    chop = np.where((range_14 == 0) | np.isnan(chop), 50.0, chop)
    
    # Align all indicators to primary timeframe (4h)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 10)  # BB(20), Chop(14), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        upper_bb = upper_bb_aligned[i]
        lower_bb = lower_bb_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        chop_val = chop_aligned[i]
        price = close[i]
        
        # Get current 4h volume for volume spike filter
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        current_vol_4h = vol_4h_aligned[i]
        
        # Volume spike filter: current 4h volume > 1.8x median volume
        volume_spike = current_vol_4h > (vol_median * 1.8)
        
        # Regime filter: chop < 61.8 indicates trending market (not choppy/ranging)
        trending = chop_val < 61.8
        
        # Breakout conditions
        breakout_long = price > upper_bb
        breakout_short = price < lower_bb
        
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
            # Breakout above upper Bollinger Band, volume spike, and trending market (chop < 61.8)
            if breakout_long and volume_spike and trending:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Breakout below lower Bollinger Band, volume spike, and trending market (chop < 61.8)
            elif breakout_short and volume_spike and trending:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dBollingerBreakout_VolumeSpike1.8x_Chop61.8_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0