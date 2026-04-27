#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams Alligator (Jaw/Teeth/Lips) with volume confirmation and 1d trend filter.
# Long when price > Alligator Lips, Lips > Teeth, Teeth > Jaw (bullish alignment) with volume > 1.5x average and 1d close > EMA50.
# Short when price < Alligator Lips, Lips < Teeth, Teeth < Jaw (bearish alignment) with volume > 1.5x average and 1d close < EMA50.
# Exit when Alligator alignment breaks or volume drops below average.
# Williams Alligator uses SMAs with specific periods: Jaw=13(8), Teeth=8(5), Lips=5(3).
# Uses 4h for execution, 12h for Alligator trend, 1d for trend filter.
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator components (SMAs with specific offsets)
    # Jaw: 13-period SMA, shifted by 8 bars
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    jaw_12h = np.roll(jaw_12h, 8)
    jaw_12h[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted by 5 bars
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)
    teeth_12h[:5] = np.nan
    
    # Lips: 5-period SMA, shifted by 3 bars
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)
    lips_12h[:3] = np.nan
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h Alligator components and 20-period volume MA
    start_idx = max(20, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Alligator alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Trend filter from 1d EMA50
        bullish_trend = price > ema50_aligned[i]
        bearish_trend = price < ema50_aligned[i]
        
        if position == 0:
            # Long: bullish Alligator alignment with volume and bullish trend
            if bullish_alignment and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bearish Alligator alignment with volume and bearish trend
            elif bearish_alignment and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks or volume drops
            if not bullish_alignment or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator alignment breaks or volume drops
            if not bearish_alignment or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0