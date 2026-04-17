#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike Filter.
Long when Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND volume > 1.5x average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND volume > 1.5x average.
Exit when Alligator alignment breaks OR volume drops below average.
Uses 1d EMA50 as trend filter: only long when price > EMA50, only short when price < EMA50.
Target: 50-150 total trades over 4 years (12-37/year).
Works in bull via Alligator alignment, works in bear via Elder Ray + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator and Elder Ray
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price
    # Jaws: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead  
    # Lips: 5-period SMMA, 3 bars ahead
    median_4h = (high_4h + low_4h) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average (Wilder's smoothing)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA*(period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws_4h = smma(median_4h, 13)
    teeth_4h = smma(median_4h, 8)
    lips_4h = smma(median_4h, 5)
    
    # Shift for Alligator's forward shift
    jaws_4h = np.roll(jaws_4h, 8)
    teeth_4h = np.roll(teeth_4h, 5)
    lips_4h = np.roll(lips_4h, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_4h = pd.Series(close_4h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_4h = high_4h - ema13_4h
    bear_power_4h = low_4h - ema13_4h
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    jaws_4h_aligned = align_htf_to_ltf(prices, df_4h, jaws_4h)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    bull_power_4h_aligned = align_htf_to_ltf(prices, df_4h, bull_power_4h)
    bear_power_4h_aligned = align_htf_to_ltf(prices, df_4h, bear_power_4h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    volume_average = volume > volume_ma  # for exit condition
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaws_4h_aligned[i]) or 
            np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or 
            np.isnan(bull_power_4h_aligned[i]) or 
            np.isnan(bear_power_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        vol_avg = volume_average[i]
        jaws = jaws_4h_aligned[i]
        teeth = teeth_4h_aligned[i]
        lips = lips_4h_aligned[i]
        bull_power = bull_power_4h_aligned[i]
        bear_power = bear_power_4h_aligned[i]
        ema50 = ema50_1d_aligned[i]
        
        # Alligator alignment
        bullish_align = jaws < teeth < lips
        bearish_align = jaws > teeth > lips
        
        if position == 0:
            # Long: bullish alignment AND Bull Power > 0 AND volume spike AND price > EMA50
            if bullish_align and bull_power > 0 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND Bear Power < 0 AND volume spike AND price < EMA50
            elif bearish_align and bear_power < 0 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment breaks OR volume drops OR price < EMA50
            if not (bullish_align and bull_power > 0) or not vol_avg or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks OR volume drops OR price > EMA50
            if not (bearish_align and bear_power < 0) or not vol_avg or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Alligator_ElderRay_VolumeSpike_EMA50Filter"
timeframe = "4h"
leverage = 1.0