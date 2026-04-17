#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ATR-based breakout with 1w ADX trend filter and volume confirmation.
# Uses daily ATR breakouts for momentum in both bull and bear markets.
# Filters with weekly ADX to only trade in trending regimes (ADX>25).
# Volume confirmation ensures breakouts have participation.
# Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for ATR breakout calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(15) on daily data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with same length
    
    atr_15 = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    atr_15_aligned = align_htf_to_ltf(prices, df_1d, atr_15)
    
    # === 1w data for ADX trend filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    # True Range
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w = np.concatenate([[np.nan], tr_w])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_w
    minus_di = 100 * minus_dm_smooth / atr_w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    # === Volume confirmation (using 1d volume) ===
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    warmup = 50  # Sufficient for all indicators
    position = 0
    
    for i in range(warmup, n):
        if (np.isnan(atr_15_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # ADX filter: only trade in trending markets (ADX > 25)
        trend_filter = adx_aligned[i] > 25
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        if position == 0:
            # Long: price breaks above highest high of past 10 days + trend + volume
            highest_high = np.max(high[max(0, i-10):i])
            if close[i] > highest_high and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lowest low of past 10 days + trend + volume
            lowest_low = np.min(low[max(0, i-10):i])
            if close[i] < lowest_low and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long on trend reversal or volatility contraction
            lowest_low = np.min(low[max(0, i-10):i])
            if close[i] < lowest_low or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on trend reversal or volatility contraction
            highest_high = np.max(high[max(0, i-10):i])
            if close[i] > highest_high or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_ATRBreakout_1wADX_VolumeFilter"
timeframe = "1d"
leverage = 1.0