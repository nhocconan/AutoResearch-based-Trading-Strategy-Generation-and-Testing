#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Weekly Trend + Volume Spike
# Uses Williams Alligator (Jaws/Teeth/Lips) on 6h for trend direction and entry signals,
# filtered by 1d EMA50 trend (bullish/bearish bias) and 1d volume spike (>2x average).
# Designed to capture trends in both bull and bear markets by following the daily trend
# while using Alligator crossovers for precise entry/exit timing. Target: 15-35 trades/year.

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume spike
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate daily volume average for volume spike detection
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Calculate Williams Alligator on 6h data (13,8,5 periods with future shifts)
    # Jaws: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        result = np.full(len(arr), np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: (prev*(period-1) + current) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply forward shifts (Jaws +8, Teeth +5, Lips +3)
    jaws_shifted = np.full_like(jaws, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaws) > 8:
        jaws_shifted[8:] = jaws[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Align daily indicators to 6h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 50, 20) + 8  # warmup for Alligator + shifts
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 6h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Alligator alignment with daily trend filter and volume spike
            # Bullish alignment: Lips > Teeth > Jaws (all rising)
            bullish_alignment = (lips_shifted[i] > teeth_shifted[i] > jaws_shifted[i])
            # Bearish alignment: Lips < Teeth < Jaws (all falling)
            bearish_alignment = (lips_shifted[i] < teeth_shifted[i] < jaws_shifted[i])
            
            # Long when bullish alignment AND daily trend bullish AND volume spike
            long_condition = (
                bullish_alignment and
                close[i] > ema50_daily_aligned[i] and   # price above daily EMA50 (bullish bias)
                vol_spike
            )
            
            # Short when bearish alignment AND daily trend bearish AND volume spike
            short_condition = (
                bearish_alignment and
                close[i] < ema50_daily_aligned[i] and   # price below daily EMA50 (bearish bias)
                vol_spike
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator death cross (Lips < Jaws) or trend reversal
            if lips_shifted[i] < jaws_shifted[i] or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator death cross (Lips > Jaws) or trend reversal
            if lips_shifted[i] > jaws_shifted[i] or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals