#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with volume confirmation and ATR stoploss.
# Long when price breaks above 1w Donchian(20) high AND volume > 1.5x median volume.
# Short when price breaks below 1w Donchian(20) low AND volume > 1.5x median volume.
# Exit when price crosses 1w Donchian(10) midline OR ATR-based stoploss triggers.
# Uses discrete position size 0.25. Targets 7-25 trades/year (30-100 total over 4 years).
# Weekly Donchian captures major trend breaks; volume confirmation filters false breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === 1w Indicators: Donchian Channels ===
    # Donchian(20) for breakout
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Donchian(10) for exit midline
    highest_10 = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (highest_10 + lowest_10) / 2
    
    # === 1w Indicators: ATR (14-period) for stoploss ===
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    true_range_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_14_1w = pd.Series(true_range_1w).rolling(window=14, min_periods=14).mean().values
    
    # === 1w Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (1d)
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    donchian_mid_10_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_10)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    vol_median_aligned = align_htf_to_ltf(prices, df_1w, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14)  # Donchian(20) needs 20, ATR needs 14
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(donchian_mid_10_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol = volume[i]
        atr = atr_14_aligned[i]
        vol_median = vol_median_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = vol > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian(10) midline OR ATR stoploss (2.0 * ATR)
            if (price < donchian_mid_10_aligned[i]) or (price < entry_price - 2.0 * atr):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian(10) midline OR ATR stoploss (2.0 * ATR)
            if (price > donchian_mid_10_aligned[i]) or (price > entry_price + 2.0 * atr):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian(20) high AND volume spike
            if (price > highest_20_aligned[i]) and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price < Donchian(20) low AND volume spike
            elif (price < lowest_20_aligned[i]) and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wDonchian20_Breakout_VolumeSpike1.5x_ATRStop2.0_MidlineExit_v1"
timeframe = "1d"
leverage = 1.0