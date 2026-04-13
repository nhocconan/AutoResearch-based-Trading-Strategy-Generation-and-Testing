#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) + 1w volume confirmation
    # Alligator identifies trend (Jaw > Teeth > Lips = uptrend, reverse = downtrend)
    # Weekly volume spike confirms institutional participation
    # Discrete sizing (0.25) minimizes fee drag. Target: 12-25 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Williams Alligator (SMMA of median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_price_1d = (high_1d + low_1d) / 2
    
    # Calculate SMMA (Smoothed Moving Average) - Wilder's smoothing
    def smma(source, period):
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Williams Alligator: Jaw (13, 8), Teeth (8, 5), Lips (5, 3)
    jaw = smma(median_price_1d, 13)
    teeth = smma(median_price_1d, 8)
    lips = smma(median_price_1d, 5)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.ones(len(df_1w))
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_avg_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period average
        idx_1w = i // (24 * 2 * 7)  # 1w bars in 12h timeframe (14 bars per week)
        if idx_1w >= len(volume_1w):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1w[idx_1w] > 1.5 * vol_avg_20_1w_aligned[i]
        
        # Alligator trend conditions
        bullish_alignment = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        bearish_alignment = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        
        # Entry conditions: Alligator alignment + volume confirmation
        enter_long = bullish_alignment and volume_confirmed
        enter_short = bearish_alignment and volume_confirmed
        
        # Stoploss: based on Alligator width ( Jaw - Lips )
        alligator_width = abs(jaw_aligned[i] - lips_aligned[i])
        stop_distance = alligator_width * 0.15  # 15% of Alligator width
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_1w_williams_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0