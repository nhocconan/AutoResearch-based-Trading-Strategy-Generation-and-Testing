#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss.
# Long when price breaks above 1w Donchian(20) upper band AND 1d volume > 2.0x 20-period average.
# Short when price breaks below 1w Donchian(20) lower band AND 1d volume > 2.0x 20-period average.
# Exit via ATR trailing stop: long stops if price < highest_high_since_entry - 2.5*ATR; short stops if price > lowest_low_since_entry + 2.5*ATR.
# Uses discrete position size 0.25. Weekly Donchian captures major trends, volume filters false breakouts, ATR stop manages risk.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 30-100 trades over 4 years (7-25/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w Indicators: Donchian Channels (20) ===
    donchian_upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_middle_20_1w = (donchian_upper_20_1w + donchian_lower_20_1w) / 2.0
    
    # === 1d Indicators: Volume average (20) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Indicators: ATR (14) for trailing stop ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_20_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state and extreme prices for trailing stop
    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_middle = donchian_middle_aligned[i]
        vol_ma = vol_ma_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === TRAILING STOP LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Check if price dropped below highest - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if price < lowest_since_entry or lowest_since_entry == 0.0:
                lowest_since_entry = price
            # Check if price rose above lowest + 2.5*ATR
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
            # LONG: Price breaks above 1w Donchian upper AND 1d volume > 2.0x 20-period avg
            if (price > donchian_upper) and (vol > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # Initialize tracking
            
            # SHORT: Price breaks below 1w Donchian lower AND 1d volume > 2.0x 20-period avg
            elif (price < donchian_lower) and (vol > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # Initialize tracking
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wDonchian20_VolumeConfirmation_ATRStop_V1"
timeframe = "1d"
leverage = 1.0