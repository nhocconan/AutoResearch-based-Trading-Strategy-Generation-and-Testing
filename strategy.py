#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
Long when price breaks above 4h Camarilla R3 level AND 12h close > 12h EMA34 (uptrend) AND volume > 2.5x 20-period MA.
Short when price breaks below 4h Camarilla S3 level AND 12h close < 12h EMA34 (downtrend) AND volume > 2.5x 20-period MA.
Exit when price retouches 4h Camarilla H5/L5 levels (mean reversion zone) or 12h trend reverses.
Camarilla levels provide precise intraday support/resistance; 12h EMA34 filters counter-trend trades; high volume threshold reduces false breakouts.
Designed for low trade frequency (target: 20-40/year) to minimize fee drag and work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Need daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar using prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: H5 = close + 1.1*(high-low)/2, L5 = close - 1.1*(high-low)/2
    # R3 = close + 1.1*(high-low)/6, S3 = close - 1.1*(high-low)/6
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H6 = close + 1.1*(high-low), L6 = close - 1.1*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h5 = close_1d + 1.1 * range_1d / 2
    camarilla_l5 = close_1d - 1.1 * range_1d / 2
    camarilla_r3 = close_1d + 1.1 * range_1d / 6
    camarilla_s3 = close_1d - 1.1 * range_1d / 6
    camarilla_h3 = close_1d + 1.1 * range_1d / 4
    camarilla_l3 = close_1d - 1.1 * range_1d / 4
    
    # Align daily Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h5_4h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_4h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or np.isnan(camarilla_h3_4h[i]) or 
            np.isnan(camarilla_l3_4h[i]) or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA34 = uptrend, close < EMA34 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_34_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_34_12h_aligned[i]
        
        # Volume filter: 4h volume > 2.5x 20-period MA (high threshold for low trade frequency)
        vol_filter = volume[i] > 2.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 level AND uptrend AND volume filter
            if close[i] > camarilla_r3_4h[i] and trend_up and vol_filter:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 level AND downtrend AND volume filter
            elif close[i] < camarilla_s3_4h[i] and trend_down and vol_filter:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retouches H3 level (mean reversion) OR 12h trend turns down
                if close[i] < camarilla_h3_4h[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retouches L3 level (mean reversion) OR 12h trend turns up
                if close[i] > camarilla_l3_4h[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0