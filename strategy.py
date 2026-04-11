#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return signals
    
    # Pre-compute Camarilla pivot levels (R3, R4, S3, S4)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    R3 = pivot + (range_12h * 1.1 / 2)
    R4 = pivot + (range_12h * 1.1)
    S3 = pivot - (range_12h * 1.1 / 2)
    S4 = pivot - (range_12h * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_12h, R3)
    R4_6h = align_htf_to_ltf(prices, df_12h, R4)
    S3_6h = align_htf_to_ltf(prices, df_12h, S3)
    S4_6h = align_htf_to_ltf(prices, df_12h, S4)
    
    # Pre-compute 6h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(R3_6h[i]) or np.isnan(R4_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Breakout conditions
        breakout_long = price_high > R4_6h[i]  # Break above R4
        breakout_short = price_low < S4_6h[i]  # Break below S4
        
        # Fade conditions (mean reversion at extreme levels)
        fade_long = price_low <= S3_6h[i] and price_close > S3_6h[i]  # Bounce from S3
        fade_short = price_high >= R3_6h[i] and price_close < R3_6h[i]  # Rejection from R3
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Breakout above R4 OR fade from S3 with volume confirmation
        if (breakout_long or fade_long) and vol_confirm:
            enter_long = True
        
        # Short: Breakdown below S4 OR fade from R3 with volume confirmation
        if (breakout_short or fade_short) and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite signal or loss of momentum
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if breakdown below S3 or fade signal at R3
            exit_long = (price_low < S3_6h[i]) or fade_short
        elif position == -1:
            # Exit short if bounce above R3 or fade signal at S3
            exit_short = (price_high > R3_6h[i]) or fade_long
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals