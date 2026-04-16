#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w EMA(50) for HTF trend direction + 12h Donchian(20) breakout with volume confirmation.
# Long when price > 1w EMA(50) AND breaks above 12h Donchian upper(20) with volume spike (>1.6x median volume).
# Short when price < 1w EMA(50) AND breaks below 12h Donchian lower(20) with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Volume spike and ATR trailing stop reduce whipsaw and overtrading.
# EMA adapts to volatility, making it effective in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data once before loop for EMA(50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Get 12h data for Donchian, volume, and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Donchian(20), Volume Median, ATR(10) ===
    # Donchian channels (20-period)
    donchian_upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
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
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 10)  # EMA(50), Donchian(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        ema_50_val = ema_50_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 12h volume for volume spike filter
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        current_vol_12h = vol_12h_aligned[i]
        
        # Volume spike filter: current 12h volume > 1.6x median volume
        volume_spike = current_vol_12h > (vol_median * 1.6)
        
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
            # Price above 1w EMA(50) (uptrend), breakout above Donchian upper, volume spike
            if price > ema_50_val and price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Price below 1w EMA(50) (downtrend), breakout below Donchian lower, volume spike
            elif price < ema_50_val and price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1wEMA50_12hDonchian20_Breakout_VolumeSpike1.6x_ATRTrail2.5_v1"
timeframe = "12h"
leverage = 1.0