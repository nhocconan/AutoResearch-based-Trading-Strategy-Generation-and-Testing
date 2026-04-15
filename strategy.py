#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend direction.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish).
# Entry confirmed by 1d EMA50 trend and volume > 1.5x median volume.
# Works in bull markets (follow Alligator up) and bear markets (follow Alligator down).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator components (all SMAs)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_12h = (high_12h + low_12h) / 2
    sma_13 = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(sma_13, 8)  # shifted 8 bars forward
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    sma_8 = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(sma_8, 5)  # shifted 5 bars forward
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    sma_5 = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(sma_5, 3)  # shifted 3 bars forward
    
    # Align indicators to 12h timeframe (already on 12h, just need to align to main timeframe)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            continue
        
        # Bullish Alligator alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish Alligator alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Volume confirmation: current volume > 1.5x median of last 20 bars
        vol_median = np.median(volume[max(0, i-20):i+1]) if i >= 20 else np.mean(volume[:i+1])
        volume_confirm = volume[i] > 1.5 * vol_median
        
        # 1d trend filter: price above/below EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Long entry: bullish alignment + volume + price above EMA50
        if bullish_alignment and volume_confirm and price_above_ema and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment + volume + price below EMA50
        elif bearish_alignment and volume_confirm and price_below_ema and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Alligator alignment or loss of volume confirmation
        elif position == 1 and (bearish_alignment or not volume_confirm):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_alignment or not volume_confirm):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0