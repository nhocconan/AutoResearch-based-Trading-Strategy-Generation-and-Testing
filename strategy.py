#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Camarilla pivot levels identify key intraday support/resistance. Breakouts above R3 or below S3
# with daily trend alignment and volume conviction capture strong momentum moves.
# Daily trend filter avoids counter-trend trades in ranging markets. Designed for 4h timeframe
# to balance trade frequency and signal quality, targeting 20-50 trades/year.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(20) for trend
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align daily EMA to 4h (changes only when daily bar closes)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Camarilla pivots from previous day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align daily Camarilla levels to 4h (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Daily EMA(20) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Camarilla R3, above daily EMA20, volume confirm
            if price > camarilla_r3_aligned[i] and price > ema_20_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < Camarilla S3, below daily EMA20, volume confirm
            elif price < camarilla_s3_aligned[i] and price < ema_20_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to daily EMA20 or below Camarilla S3
            if price < ema_20_1d_aligned[i] or price < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to daily EMA20 or above Camarilla R3
            if price > ema_20_1d_aligned[i] or price > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals