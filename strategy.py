#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# Volume spike confirms breakout authenticity; chop filter avoids whipsaws in ranging markets
# Works in bull/bear: Alligator adapts to trend, volume confirms momentum, chop filter reduces false signals
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Alligator and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)  # Shift 8 bars forward
    teeth = np.roll(teeth, 5)  # Shift 5 bars forward
    lips = np.roll(lips, 3)  # Shift 3 bars forward
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align 1d Alligator to 12h timeframe (wait for 1d bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Choppiness Index on 1d (to filter ranging markets)
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR for chop
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr = true_range(df_1d['high'].iloc[i], df_1d['low'].iloc[i], df_1d['close'].iloc[i-1])
        if i < 14:
            atr_1d[i] = np.nan
        else:
            if i == 14:
                atr_1d[i] = np.mean(true_range(
                    df_1d['high'].iloc[1:15].values,
                    df_1d['low'].iloc[1:15].values,
                    np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[1:14].values])
                ))
            else:
                atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    # Choppiness Index = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if np.isnan(atr_1d[i]):
            chop_1d[i] = np.nan
            continue
        # Sum of ATR over last 14 periods
        atr_sum = np.sum(atr_1d[i-13:i+1])
        # Max high and min low over last 14 periods
        max_high = np.max(df_1d['high'].iloc[i-13:i+1].values)
        min_low = np.min(df_1d['low'].iloc[i-13:i+1].values)
        if max_high == min_low:
            chop_1d[i] = 0
        else:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Align chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period average volume on 1d for volume confirmation
    avg_volume_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 20:
            avg_volume_1d[i] = np.nan
        else:
            avg_volume_1d[i] = np.mean(df_1d['volume'].iloc[i-20:i].values)
    
    # Align volume average to 12h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (scaled to 12h)
        # Need to get 1d volume at this point - approximate by using current 12h volume vs daily average
        # Simpler approach: use volume ratio directly
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Chop regime filter: only trade when market is trending (CHOP < 38.2) or extreme ranging (CHOP > 61.8)
        # For Alligator strategy, we prefer trending markets
        chop_filter = chop_1d_aligned[i] < 38.2  # Trending market regime
        
        if position == 1:  # Long position
            # Exit: Alligator lines cross (lips below teeth) OR chop enters ranging market
            if lips_aligned[i] < teeth_aligned[i] or chop_1d_aligned[i] > 50.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (lips above teeth) OR chop enters ranging market
            if lips_aligned[i] > teeth_aligned[i] or chop_1d_aligned[i] > 50.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Alligator alignment + volume confirmation + chop filter
            if volume_confirmed and chop_filter:
                # Strong uptrend: lips > teeth > jaw AND price above lips
                if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                    close[i] > lips_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Strong downtrend: lips < teeth < jaw AND price below lips
                elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                      close[i] < lips_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals