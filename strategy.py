#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Supertrend for HTF trend bias + 4h Donchian(20) breakout with volume confirmation.
# Long when 1d Supertrend is bullish AND price breaks above 4h Donchian upper band with volume spike (>2.0x median volume).
# Short when 1d Supertrend is bearish AND price breaks below 4h Donchian lower band with volume spike.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.30. 1d Supertrend filters for higher-timeframe trend direction,
# Donchian breakout captures momentum, volume confirmation reduces false signals.
# Designed to work in both bull and bear markets by following the dominant 1d trend with precise 4h entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Supertrend(10,3.0) ===
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upperband = hl2 + (multiplier * atr_1d)
    lowerband = hl2 - (multiplier * atr_1d)
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[atr_period-1] = upperband[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close_1d)):
        if close_1d[i] <= supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = 1
        
        if direction[i] == 1:
            supertrend[i] = max(upperband[i], supertrend[i-1])
        else:
            supertrend[i] = min(lowerband[i], supertrend[i-1])
    
    # Supertrend is bullish when direction = 1, bearish when direction = -1
    supertrend_bullish = (direction == 1)
    supertrend_bearish = (direction == -1)
    
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
    supertrend_bullish_aligned = align_htf_to_ltf(prices, df_1d, supertrend_bullish.astype(float))
    supertrend_bearish_aligned = align_htf_to_ltf(prices, df_1d, supertrend_bearish.astype(float))
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(10, 20, 10)  # Supertrend(10), Donchian(20), ATR(10)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_bullish_aligned[i]) or np.isnan(supertrend_bearish_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        st_bullish = supertrend_bullish_aligned[i] > 0.5
        st_bearish = supertrend_bearish_aligned[i] > 0.5
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        
        # Get current 4h volume for volume spike filter
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        current_vol_4h = vol_4h_aligned[i]
        
        # Volume spike filter: current 4h volume > 2.0x median volume
        volume_spike = current_vol_4h > (vol_median * 2.0)
        
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
            # 1d Supertrend bullish, price breaks above Donchian upper, volume spike
            if st_bullish and price > upper and volume_spike:
                signals[i] = 0.30
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # 1d Supertrend bearish, price breaks below Donchian lower, volume spike
            elif st_bearish and price < lower and volume_spike:
                signals[i] = -0.30
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.30  # maintain position
    
    return signals

name = "4h_1dSupertrend10_3_4hDonchian20_Breakout_VolumeSpike2.0x_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0