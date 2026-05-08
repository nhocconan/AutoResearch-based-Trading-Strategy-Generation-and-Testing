#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1d Williams Alligator + Volume Spike + Chop Filter
# Uses daily Williams Alligator (3 SMAs) for trend direction, volume > 2x average for entry,
# and chop filter (Choppiness Index > 61.8) to avoid sideways markets. Designed to capture
# strong trends in both bull and bear markets while avoiding whipsaws in chop.
# Target: 12-37 trades/year on 12h timeframe.

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and chop filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5)
    close_daily = df_daily['close'].values
    
    # Jaw (13-period SMA)
    jaw = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 13:
        for i in range(12, len(close_daily)):
            jaw[i] = np.mean(close_daily[i-12:i+1])
    
    # Teeth (8-period SMA)
    teeth = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 8:
        for i in range(7, len(close_daily)):
            teeth[i] = np.mean(close_daily[i-7:i+1])
    
    # Lips (5-period SMA)
    lips = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 5:
        for i in range(4, len(close_daily)):
            lips[i] = np.mean(close_daily[i-4:i+1])
    
    # Calculate daily Choppiness Index (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 2:
        tr[1:] = np.maximum(high_daily[1:] - low_daily[1:], 
                           np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                                      np.abs(low_daily[1:] - close_daily[:-1])))
    
    # ATR (14-period SMA of TR)
    atr_14 = np.full(len(close_daily), np.nan)
    if len(tr) >= 15:  # need at least 14 TR values + 1 for index shift
        # Initialize first ATR as average of first 14 TR values
        atr_14[14] = np.nanmean(tr[1:15])
        for i in range(15, len(tr)):
            if np.isnan(atr_14[i-1]):
                atr_14[i] = np.nanmean(tr[i-13:i+1])
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = np.full(len(close_daily), np.nan)
    lowest_low_14 = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        for i in range(13, len(close_daily)):
            highest_high_14[i] = np.max(high_daily[i-13:i+1])
            lowest_low_14[i] = np.min(low_daily[i-13:i+1])
    
    # Choppiness Index: CI = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        for i in range(14, len(close_daily)):
            if not np.isnan(atr_14[i]) and not np.isnan(highest_high_14[i]) and not np.isnan(lowest_low_14[i]):
                if highest_high_14[i] > lowest_low_14[i]:
                    sum_atr = np.nansum(atr_14[i-13:i+1])
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((highest_high_14[i] - lowest_low_14[i]) + 1e-10)
                else:
                    chop[i] = 50  # neutral when no range
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20 = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(19, len(vol_daily)):
            vol_avg_20[i] = np.mean(vol_daily[i-19:i+1])
    
    # Align daily indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 14, 19)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 12h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Alligator alignment: Jaw > Teeth > Lips = bullish, Jaw < Teeth < Lips = bearish
        bullish_aligned = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        bearish_aligned = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        if position == 0:
            # Look for entry: Alligator aligned in non-choppy market with volume spike
            # Chop < 38.2 indicates trending market (good for trend following)
            trending_market = chop_aligned[i] < 38.2
            
            # Long when Alligator bullish aligned
            long_condition = bullish_aligned and trending_market and vol_spike
            
            # Short when Alligator bearish aligned
            short_condition = bearish_aligned and trending_market and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish or market becomes choppy
            if not bullish_aligned or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish or market becomes choppy
            if not bearish_aligned or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals