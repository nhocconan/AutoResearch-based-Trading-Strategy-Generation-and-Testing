#!/usr/bin/env python3
# 1d_WilliamsAlligator_1wTrend
# Hypothesis: Use Williams Alligator (Jaw, Teeth, Lips) on weekly timeframe to determine trend direction,
# filtered by daily price position relative to the Alligator's Teeth (SMMA8) and volume confirmation.
# Go long when price > Teeth and Lips > Jaw (bullish alignment), short when price < Teeth and Lips < Jaw (bearish alignment).
# Exit on opposite signal or when price crosses Jaw. Designed for low frequency (<25 trades/year) to avoid fee drag.
# Williams Alligator uses smoothed moving averages (SMMA) which are less noisy and better for trend identification.

name = "1d_WilliamsAlligator_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smoothed_moving_average(data, period):
    """
    Calculate Smoothed Moving Average (SMMA).
    SMMA is similar to EMA but with different smoothing factor.
    SMMA today = (SMMA yesterday * (period-1) + price today) / period
    """
    n = len(data)
    smma = np.full(n, np.nan)
    if n == 0:
        return smma
    # Initialize with SMA for first value
    smma[0] = np.mean(data[:min(period, n)])
    for i in range(1, n):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def williams_alligator(high, low, close):
    """
    Calculate Williams Alligator indicator.
    Returns Jaw (SMMA13), Teeth (SMMA8), Lips (SMMA5) of median price.
    """
    median_price = (high + low) / 2
    jaw = smoothed_moving_average(median_price, 13)  # Jaw: Blue line (13-period)
    teeth = smoothed_moving_average(median_price, 8)  # Teeth: Red line (8-period)
    lips = smoothed_moving_average(median_price, 5)   # Lips: Green line (5-period)
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Alligator trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Alligator on weekly data
    jaw_1w, teeth_1w, lips_1w = williams_alligator(high_1w, low_1w, close_1w)
    
    # Daily EMA for volatility filter (optional)
    ema_50_d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Alligator lines to daily timeframe
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(lips_1w_aligned[i]) or 
            np.isnan(ema_50_d[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: Alligator alignment
        # Bullish: Lips > Teeth > Jaw (green above red above blue)
        bullish_alignment = lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i]
        # Bearish: Lips < Teeth < Jaw (green below red below blue)
        bearish_alignment = lips_1w_aligned[i] < teeth_1w_aligned[i] < jaw_1w_aligned[i]
        
        # Price relative to Teeth (8-period SMMA)
        price_above_teeth = close[i] > teeth_1w_aligned[i]
        price_below_teeth = close[i] < teeth_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Bullish alignment AND price above Teeth AND volume confirmation
            if bullish_alignment and price_above_teeth and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment AND price below Teeth AND volume confirmation
            elif bearish_alignment and price_below_teeth and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bearish alignment OR price crosses below Jaw
            if bearish_alignment or close[i] < jaw_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish alignment OR price crosses above Jaw
            if bullish_alignment or close[i] > jaw_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals