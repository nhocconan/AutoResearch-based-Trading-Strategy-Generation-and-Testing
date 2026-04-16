#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 4h Donchian(20) breakout with 1d volume spike confirmation and ATR-based stoploss.
# Long when price breaks above 20-period Donchian high and 1d volume > 1.5x 20-period median volume.
# Short when price breaks below 20-period Donchian low and same volume condition.
# Exit via ATR(14) trailing stop: long exits when price < highest high since entry - 2.0*ATR,
# short exits when price > lowest low since entry + 2.0*ATR.
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian breakouts capture strong momentum moves; volume confirmation filters false breakouts;
# ATR trailing stop adapts to volatility and reduces whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian(20) channels and ATR(14) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian(20) channels
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # ATR(14)
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2.iloc[0] = tr1.iloc[0]  # first bar: no previous close
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: volume median ===
    volume_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20, 14, 20)  # Donchian(20), ATR(14), volume median(20)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr = atr_aligned[i]
        vol_median = vol_median_aligned[i]
        price = close[i]
        
        # Get current 1d volume for volume spike filter
        df_1d = get_htf_data(prices, '1d')
        vol_1d = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
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
            # Price breaks above Donchian high and volume spike
            if price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Price breaks below Donchian low and volume spike
            elif price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_1dVolumeSpike1.5x_ATRTrail2.0_v1"
timeframe = "4h"
leverage = 1.0