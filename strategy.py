#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band squeeze breakout with volume confirmation and daily trend filter.
# Uses weekly Bollinger Bands to detect low volatility (squeeze) conditions.
# Enters on daily breakouts above/below the weekly Bollinger Bands with volume confirmation.
# Trend filter uses daily EMA20 to ensure trades align with higher timeframe momentum.
# Designed for low frequency (target 10-30 trades/year) to minimize fee drag in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate basis (SMA)
    basis = np.full(len(close_1w), np.nan)
    for i in range(bb_length, len(close_1w)):
        basis[i] = np.mean(close_1w[i-bb_length:i])
    
    # Calculate standard deviation
    dev = np.full(len(close_1w), np.nan)
    for i in range(bb_length, len(close_1w)):
        dev[i] = np.std(close_1w[i-bb_length:i])
    
    # Calculate upper and lower bands
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    
    # Squeeze condition: band width < 50th percentile of band width (using 50-period lookback)
    width = upper - lower
    width_percentile = np.full(len(width), np.nan)
    for i in range(50, len(width)):
        if i >= 50:
            width_percentile[i] = np.percentile(width[i-50:i], 50)
    
    squeeze = width < width_percentile
    
    # Align weekly data to daily timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(20)
    ema_20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_20_1d[19] = np.mean(close_1d[:20])  # Simple average for first value
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] * 2/21) + (ema_20_1d[i-1] * (1 - 2/21))
    
    # Align daily EMA to daily timeframe (no change needed, but for consistency)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need weekly BB data, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(squeeze_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below daily EMA20
        trend_up = close[i] > ema_20_1d_aligned[i]
        trend_down = close[i] < ema_20_1d_aligned[i]
        
        if position == 0:
            # Long entry: price above weekly upper band, with squeeze, volume and trend filter
            if (close[i] > upper_aligned[i] and 
                squeeze_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly lower band, with squeeze, volume and trend filter
            elif (close[i] < lower_aligned[i] and 
                  squeeze_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below weekly lower band or opposite squeeze breakout
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly upper band or opposite squeeze breakout
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBB20_2_Squeeze_Breakout_Volume_EMA20"
timeframe = "1d"
leverage = 1.0