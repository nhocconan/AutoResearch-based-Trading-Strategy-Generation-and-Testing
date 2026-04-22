#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h Trend Filter + Volume Spike
# Long when price > Alligator Teeth (Jaw) + 12h EMA50 up + volume spike
# Short when price < Alligator Teeth + 12h EMA50 down + volume spike
# Exit when price crosses Alligator Lips or trend reverses
# Williams Alligator: Jaw (SMMA13, offset8), Teeth (SMMA8, offset5), Lips (SMMA5, offset3)
# Designed for low frequency (~20-30/year) with trend-following edge in both bull and bear markets

def smma(source, length, offset):
    """Smoothed Moving Average (SMMA) with offset"""
    sma = pd.Series(source).rolling(window=length, min_periods=length).mean()
    # SMMA is like EMA but with alpha = 1/length
    smma_vals = np.full_like(source, np.nan, dtype=float)
    if len(source) >= length:
        smma_vals[length-1] = sma[length-1]
        for i in range(length, len(source)):
            smma_vals[i] = (smma_vals[i-1] * (length-1) + source[i]) / length
    return np.roll(smma_vals, -offset)  # negative offset shifts left (future)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Williams Alligator and trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams Alligator components (using median price)
    median_price = (high_12h + low_12h) / 2
    
    # Jaw: SMMA(13, offset8)
    jaw = smma(median_price, 13, 8)
    # Teeth: SMMA(8, offset5)  
    teeth = smma(median_price, 8, 5)
    # Lips: SMMA(5, offset3)
    lips = smma(median_price, 5, 3)
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h EMA50 slope for trend direction
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    
    # Calculate 24-period average volume for volume spike detection (using 12h volume scaled to 6h)
    # Since we're on 6h chart, use 24 periods of 6h volume ≈ 12 periods of 12h volume
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_50_aligned[i]
        ema_slope = ema_50_slope[i]
        
        # Volume filter: current volume > 2.0 * 24-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Alligator alignment: price > Teeth = bullish alignment, price < Teeth = bearish
        bullish_alignment = price > teeth_val
        bearish_alignment = price < teeth_val
        
        if position == 0:
            # Long conditions: bullish alignment + up trend + volume spike
            if bullish_alignment and ema_slope > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + down trend + volume spike
            elif bearish_alignment and ema_slope < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Lips or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below Lips or trend turns down
                if price < lips_val or ema_slope <= 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above Lips or trend turns up
                if price > lips_val or ema_slope >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0