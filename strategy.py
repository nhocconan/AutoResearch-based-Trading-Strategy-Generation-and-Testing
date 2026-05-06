#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and volume confirmation
# Uses Williams Alligator (SMAs with specific offsets) to identify trend direction and strength
# 1w EMA50 for higher timeframe trend alignment (reduces whipsaw in ranging markets)
# Volume spike (>1.8x 30-bar average) confirms breakout strength
# ATR-based stoploss via signal=0 when price crosses opposite Alligator line
# Discrete sizing 0.25 to limit fee drag; target 50-150 total trades over 4 years (12-37/year)
# Williams Alligator is effective in both trending and ranging markets - works in bull/bear

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    close_1d_series = pd.Series(close_1d)
    jaw = smma(close_1d_series.values, 13)
    teeth = smma(close_1d_series.values, 8)
    lips = smma(close_1d_series.values, 5)
    
    # Shift as per Alligator definition: Jaw(8), Teeth(5), Lips(3)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Fill rolled values with NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate volume spike filter (>1.8x 30-bar average on 1d)
    vol_ma_30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume_1d > (1.8 * vol_ma_30)
    
    # Align HTF indicators to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up AND price > lips AND uptrend (price > EMA50_1w) AND volume spike
            if (lips_above_teeth and teeth_above_jaw and 
                close[i] > lips_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND price < lips AND downtrend (price < EMA50_1w) AND volume spike
            elif (lips_below_teeth and teeth_below_jaw and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns down (lips < teeth) OR price retests teeth from above
            if lips_aligned[i] < teeth_aligned[i] or close[i] <= teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns up (lips > teeth) OR price retests teeth from below
            if lips_aligned[i] > teeth_aligned[i] or close[i] >= teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals