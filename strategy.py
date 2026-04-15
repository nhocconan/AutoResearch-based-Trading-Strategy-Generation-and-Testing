#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX trend strength + 12h EMA crossover + volume confirmation
# Uses ADX to filter for trending markets, EMA crossover for entry signals,
# and volume to confirm momentum. Works in both bull and bear by only trading
# when ADX > 25 (strong trend). Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Load 12h data for EMA crossover
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 4h
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14 * 100
    minus_di_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14 * 100
    
    # DX and ADX
    dx = np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate EMA crossover on 12h (9 and 21 periods)
    ema9_12h = pd.Series(close_12h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume average (20-period on 4h)
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    ema9_12h_aligned = align_htf_to_ltf(prices, df_12h, ema9_12h)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(ema9_12h_aligned[i]) or
            np.isnan(ema21_12h_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: ADX > 25 (trending) + EMA9 crosses above EMA21 + volume confirmation
        if (adx_aligned[i] > 25 and
            ema9_12h_aligned[i] > ema21_12h_aligned[i] and
            volume[i] > 1.3 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: ADX > 25 (trending) + EMA9 crosses below EMA21 + volume confirmation
        elif (adx_aligned[i] > 25 and
              ema9_12h_aligned[i] < ema21_12h_aligned[i] and
              volume[i] > 1.3 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: ADX < 20 (weak trend) or reverse EMA crossover
        elif position == 1 and (adx_aligned[i] < 20 or ema9_12h_aligned[i] < ema21_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_aligned[i] < 20 or ema9_12h_aligned[i] > ema21_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_ADX_EMA_Volume_Trend"
timeframe = "4h"
leverage = 1.0