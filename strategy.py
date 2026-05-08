#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level (S3/R3) breakout with 1d trend filter and volume spike.
# Long when price closes above R3 (1d) AND 1d EMA34 rising AND volume > 1.8x 20-period average.
# Short when price closes below S3 (1d) AND 1d EMA34 falling AND volume > 1.8x 20-period average.
# Exit when price crosses back inside the S3-R3 range.
# Camarilla levels provide high-probability reversal/breakout points. The 1d EMA34 filter ensures alignment with higher timeframe trend.
# Volume spike confirms institutional participation. Designed for low trade frequency (<40/year) to minimize fee drag.

name = "4h_Camarilla_R3_S3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (S3, R3) from previous 1d candle
    # Camarilla: H-L = range, then S3 = C - (H-L)*1.1/4, R3 = C + (H-L)*1.1/4
    # Using previous day's OHLC to avoid look-ahead
    range_1d = high_1d - low_1d
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    
    # Shift by 1 to use previous day's levels (available at open)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3[0] = np.nan  # First value invalid
    camarilla_r3[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction (using aligned values)
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price closes above R3, 1d EMA34 rising, volume filter
            long_cond = (close[i] > camarilla_r3_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price closes below S3, 1d EMA34 falling, volume filter
            short_cond = (close[i] < camarilla_s3_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes back below S3 (mean reversion)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes back above R3 (mean reversion)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals