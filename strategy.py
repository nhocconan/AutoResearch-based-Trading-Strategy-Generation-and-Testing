#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Enters long when price > Alligator Jaw (blue line) AND Lips > Teeth > Jaw (bullish alignment) AND weekly trend up.
# Enters short when price < Alligator Jaw AND Lips < Teeth < Jaw (bearish alignment) AND weekly trend down.
# Uses smoothed SMAs (5,8,13) with specific offsets to avoid look-ahead. Designed for low turnover (target: 10-30 trades/year).
# Works in bull markets (trend continuation) and bear markets (trend continuation down).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with offsets)
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    def smma(series, period):
        """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
        sma = np.full_like(series, np.nan)
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply shifts (offsets) to avoid look-ahead
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    # Jaw shifted 8 bars forward
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    # Teeth shifted 5 bars forward
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    # Lips shifted 3 bars forward
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * 0.038) + (ema50_1w[i-1] * 0.962)  # alpha = 2/(50+1)
    
    # Align indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        volume_ma20[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            volume_ma20[i] = (volume[i] * 0.095) + (volume_ma20[i-1] * 0.905)  # alpha = 2/(20+1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Price relative to Jaw
        price_above_jaw = close[i] > jaw_aligned[i]
        price_below_jaw = close[i] < jaw_aligned[i]
        
        # Weekly trend filter: price above/below weekly EMA50
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: Bullish alignment + price above jaw + weekly uptrend + volume
            if (bullish_alignment and price_above_jaw and weekly_uptrend and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below jaw + weekly downtrend + volume
            elif (bearish_alignment and price_below_jaw and weekly_downtrend and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish alignment OR price crosses below jaw OR weekly trend turns down
            if bearish_alignment or (close[i] < jaw_aligned[i]) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish alignment OR price crosses above jaw OR weekly trend turns up
            if bullish_alignment or (close[i] > jaw_aligned[i]) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0