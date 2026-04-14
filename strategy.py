#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and ADX trend filter
# - Uses previous day's Camarilla pivot levels (S1/S2/S3, R1/R2/R3) for mean reversion
# - Requires volume > 1.5x 24-period average for institutional confirmation
# - Filters for trending markets using ADX > 25 to avoid ranging whipsaws
# - Designed to capture mean reversion in both bull and bear markets with controlled frequency
# - Target: 50-150 trades over 4 years to minimize fee drag while capturing significant reversals
# - Discrete position sizing (0.25) to reduce churn and manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ADX for trend filtering
    # True Range
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Calculate for each day
    camarilla_s1 = np.zeros(len(df_1d))
    camarilla_s2 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_r1 = np.zeros(len(df_1d))
    camarilla_r2 = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        range_hl = h - l
        
        camarilla_s1[i] = c - (range_hl * 1.1 / 12)
        camarilla_s2[i] = c - (range_hl * 1.1 / 6)
        camarilla_s3[i] = c - (range_hl * 1.1 / 4)
        camarilla_r1[i] = c + (range_hl * 1.1 / 12)
        camarilla_r2[i] = c + (range_hl * 1.1 / 6)
        camarilla_r3[i] = c + (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_12h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_12h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 24-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(adx_12h[i]) or np.isnan(camarilla_s1_12h[i]) or np.isnan(camarilla_r1_12h[i]) or \
           np.isnan(vol_ma[i]):
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_12h[i] <= 25:
            # If we have a position, exit when trend weakens
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Mean reversion entries:
            # Long when price touches or goes below S1 with volume confirmation
            # Short when price touches or goes above R1 with volume confirmation
            if (low[i] <= camarilla_s1_12h[i] and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            elif (high[i] >= camarilla_r1_12h[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit long: price reaches R1 (mean reversion target) or S2 (stop)
            if high[i] >= camarilla_r1_12h[i] or low[i] <= camarilla_s2_12h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: price reaches S1 (mean reversion target) or R2 (stop)
            if low[i] <= camarilla_s1_12h[i] or high[i] >= camarilla_r2_12h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Pivot_MeanReversion_Volume_ADX"
timeframe = "12h"
leverage = 1.0