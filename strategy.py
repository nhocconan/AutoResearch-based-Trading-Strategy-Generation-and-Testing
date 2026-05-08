#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Volume Spike + Chop Filter
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) on 4h to detect trend presence.
# Enters long when Lips > Teeth > Jaw (bullish alignment) with volume spike (>2x 20-period average)
# and choppy market filter (Chop > 61.8 for mean reversion context).
# Exits when alignment breaks or volatility drops.
# Designed to capture trend continuation moves in both bull and bear markets with controlled frequency.

name = "4h_WilliamsAlligator_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Chop calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate Chop (Choppiness Index) on daily
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    atr_daily = np.zeros(len(close_daily))
    for i in range(1, len(close_daily)):
        tr = max(high_daily[i] - low_daily[i],
                 abs(high_daily[i] - close_daily[i-1]),
                 abs(low_daily[i] - close_daily[i-1]))
        atr_daily[i] = tr
    
    # Smoothed ATR (using simple mean for simplicity, equivalent to Wilder's smoothing with enough data)
    atr_sum = np.zeros(len(close_daily))
    if len(close_daily) >= 14:
        atr_sum[13] = np.sum(atr_daily[1:15])  # sum of first 14 TR values
        for i in range(14, len(close_daily)):
            atr_sum[i] = atr_sum[i-1] - atr_sum[i-1]/14 + atr_daily[i]
    
    atr_14 = np.zeros(len(close_daily))
    if len(close_daily) >= 14:
        atr_14[13] = atr_sum[13] / 14
        for i in range(14, len(close_daily)):
            atr_14[i] = (atr_14[i-1] * 13 + atr_daily[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.zeros(len(close_daily))
    lowest_low = np.zeros(len(close_daily))
    for i in range(len(close_daily)):
        if i < 14:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high_daily[i-13:i+1])
            lowest_low[i] = np.min(low_daily[i-13:i+1])
    
    chop = np.full(len(close_daily), np.nan)
    for i in range(14, len(close_daily)):
        if atr_14[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(atr_14[i] * 14 / (highest_high[i] - lowest_low[i])) / np.log10(10)
    
    # Chop > 61.8 indicates ranging/choppy market
    
    # Calculate Williams Alligator on 4h data
    # Jaw: 13-period SMMA (smoothed moving average) of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        sma = np.full(len(arr), np.nan)
        if len(arr) < period:
            return sma
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Shift forward: Jaw +8, Teeth +5, Lips +3
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    for i in range(len(jaw)):
        if i + 8 < len(jaw):
            jaw[i + 8] = jaw_raw[i]
        if i + 5 < len(teeth):
            teeth[i + 5] = teeth_raw[i]
        if i + 3 < len(lips):
            lips[i + 3] = lips_raw[i]
    
    # Align daily Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop, additional_delay_bars=0)
    
    # Calculate 4h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13+8, 8+5, 5+3)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or \
           np.isnan(chop_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        chop_high = chop_aligned[i] > 61.8  # choppy/ranging market
        
        if position == 0:
            # Look for entry: Alligator alignment with volume spike in choppy market
            if bullish_alignment and vol_spike and chop_high:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and vol_spike and chop_high:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment breaks or volatility drops
            if not bullish_alignment or not vol_spike or not chop_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment breaks or volatility drops
            if not bearish_alignment or not vol_spike or not chop_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals