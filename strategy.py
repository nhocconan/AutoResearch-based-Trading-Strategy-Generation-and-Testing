#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x average volume AND choppy market (CHOP > 61.8).
# Short when price breaks below Donchian(20) low AND volume > 1.5x average volume AND choppy market (CHOP > 61.8).
# Exit when price touches Donchian midpoint or opposite channel touch.
# Uses discrete position size 0.25. Volume confirmation ensures breakout validity.
# Chop filter (>61.8) ensures we only trade in ranging markets where mean reversion works.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (catch uptrend breaks) and bear markets (catch downtrend breaks) during ranging phases.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Indicators calculated on primary timeframe ===
    # Donchian Channel (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume average (20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14)
    # CHOP = 100 * log10(sum(ATR(1) over 14) / (log10(highest high - lowest low over 14)))
    atr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr[0] = high[0] - low[0]  # first ATR
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (hh14 - ll14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((hh14 - ll14) > 0, chop, 100)  # set to 100 when range is zero
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 30  # Donchian(20) and CHOP(14) need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        vol_average = vol_ma[i]
        chop_value = chop[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        mid_channel = donchian_mid[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price touches Donchian midpoint OR breaks below lower channel
            if (price <= mid_channel) or (price < lower_channel):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price touches Donchian midpoint OR breaks above upper channel
            if (price >= mid_channel) or (price > upper_channel):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirmed = vol > (1.5 * vol_average)
            # Chop filter: choppy market (CHOP > 61.8) for mean reversion
            chop_filter = chop_value > 61.8
            
            # LONG: price breaks above upper channel AND volume confirmed AND choppy market
            if (price > upper_channel) and volume_confirmed and chop_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below lower channel AND volume confirmed AND choppy market
            elif (price < lower_channel) and volume_confirmed and chop_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_ChoppinessFilter_V1"
timeframe = "4h"
leverage = 1.0