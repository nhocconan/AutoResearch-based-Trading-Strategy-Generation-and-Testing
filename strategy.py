# State your hypothesis in a comment at the top
# Hypothesis: 12-hour timeframe strategy combining Camarilla pivot levels (from 1-day timeframe) with volume confirmation and trend filter (1-week EMA).
# The strategy enters long when price breaks above Camarilla R3 level with volume confirmation and price above weekly EMA.
# Enters short when price breaks below Camarilla S3 level with volume confirmation and price below weekly EMA.
# Uses discrete position sizing (0.25) to limit trade frequency and manage risk. Designed for low trade frequency (~12-37 trades/year) to avoid fee drag.
# Works in both bull and bear markets by using the weekly EMA as a trend filter and Camarilla levels for mean reversion/breakout structure.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot levels (HIGH, LOW, CLOSE from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using previous day's HLC
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    # We shift by 1 to use previous day's levels for current day's trading
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12-hour timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1-week data for trend filter (EMA 10)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA (on 12h timeframe)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R3 + volume confirmation + price > weekly EMA
            if (price > camarilla_r3_aligned[i] and vol_confirm[i] and price > ema_10_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S3 + volume confirmation + price < weekly EMA
            elif (price < camarilla_s3_aligned[i] and vol_confirm[i] and price < ema_10_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below Camarilla R3
            if price < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above Camarilla S3
            if price > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals