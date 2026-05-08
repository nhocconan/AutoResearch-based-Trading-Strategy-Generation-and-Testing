#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA Trend Filter + Volume Spike
# Uses 12h Williams Alligator (JAW, TEETH, LIPS) for trend direction,
# 1d EMA34 for higher timeframe trend filter, and volume spike (>2x average)
# for entry timing. Designed to work in both bull and bear markets by following
# the daily trend while avoiding choppy conditions. Target: 12-37 trades/year.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
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
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # SMMA (Smoothed Moving Average) function
    def smma(arr, period):
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            result[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA for different periods
    smma13 = smma(close_12h, 13)  # Jaw
    smma8 = smma(close_12h, 8)    # Teeth
    smma5 = smma(close_12h, 5)    # Lips
    
    # Calculate Alligator lines with shift
    jaw = np.roll(smma13, 8)   # Jaw: 13-period SMMA shifted 8 bars
    teeth = np.roll(smma8, 5)  # Teeth: 8-period SMMA shifted 5 bars
    lips = np.roll(smma5, 3)   # Lips: 5-period SMMA shifted 3 bars
    
    # Calculate daily volume average for volume spike
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 12h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Align 12h Alligator lines to 12h timeframe (no additional alignment needed as already on 12h)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 12h volume > 2x 20-period average of daily volume
        vol_spike = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_12h_current = volume[i]
            vol_spike = vol_12h_current > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Alligator aligned (Lips > Teeth > Jaw for long, Lips < Teeth < Jaw for short)
            # in alignment with daily EMA trend and volume spike
            
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            
            # Long when bullish alignment, price above daily EMA34, and volume spike
            long_condition = (
                bullish_alignment and
                close[i] > ema34_daily_aligned[i] and
                vol_spike
            )
            
            # Short when bearish alignment, price below daily EMA34, and volume spike
            short_condition = (
                bearish_alignment and
                close[i] < ema34_daily_aligned[i] and
                vol_spike
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish or price crosses below EMA34
            bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            if bearish_alignment or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish or price crosses above EMA34
            bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            if bullish_alignment or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals