#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Williams %R for HTF mean-reversion + 12h Bollinger Band squeeze breakout with volume confirmation.
# Long when 1w Williams %R < -80 (oversold) AND price breaks above 12h BB upper with volume spike (>1.8x median volume).
# Short when 1w Williams %R > -20 (overbought) AND price breaks below 12h BB lower with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.0*ATR,
# short exits when price > lowest low since entry + 2.0*ATR.
# Uses discrete position size 0.25. Williams %R identifies mean-reversion extremes in 1w timeframe,
# Bollinger squeeze breakout captures volatility expansion, volume confirmation reduces false signals.
# Designed to work in both bull and bear markets by trading mean-reversion extremes with momentum confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data once before loop for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Williams %R(14) ===
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w + 1e-10)
    
    # Get 12h data for Bollinger Bands, volume, and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Bollinger Bands(20,2), Volume Median, ATR(10) ===
    # Bollinger Bands (20-period, 2 std dev)
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Volume median (20-period)
    vol_median_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    
    # ATR(10) for trailing stop
    tr1_12h = pd.Series(high_12h - low_12h)
    tr2_12h = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3_12h = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr2_12h.iloc[0] = tr1_12h.iloc[0]
    tr3_12h.iloc[0] = tr1_12h.iloc[0]
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_10 = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to primary timeframe (12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20, 10)  # Williams %R(14), BB(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        williams_r = williams_r_aligned[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 12h volume for volume spike filter
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        current_vol_12h = vol_12h_aligned[i]
        
        # Volume spike filter: current 12h volume > 1.8x median volume
        volume_spike = current_vol_12h > (vol_median * 1.8)
        
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
            # 1w Williams %R oversold (< -80), price breaks above BB upper, volume spike
            if williams_r < -80 and price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # 1w Williams %R overbought (> -20), price breaks below BB lower, volume spike
            elif williams_r > -20 and price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1wWilliamsR14_12hBB20_Breakout_VolumeSpike1.8x_ATRTrail2.0_v1"
timeframe = "12h"
leverage = 1.0