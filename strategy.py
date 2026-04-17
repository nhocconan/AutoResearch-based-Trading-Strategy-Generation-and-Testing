#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and Donchian
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily Donchian(20) channel
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donch_high_20 = high_1d_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_1d_series.rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Weekly ADX(14) for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate directional movement
    up = high_1w[1:] - high_1w[:-1]
    down = low_1w[:-1] - low_1w[1:]
    up = np.concatenate([[0], up])
    down = np.concatenate([down, [0]])
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # True range for weekly
    high_low_w = high_1w - low_1w
    high_close_w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(high_low_w, np.maximum(high_close_w, low_close_w))
    tr_w[0] = high_low_w[0]
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Directional indicators
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_w
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Weekly close EMA(21) for trend direction
    close_1w_series = pd.Series(close_1w)
    ema21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above daily Donchian high with strong weekly uptrend
            if (close[i] > donch_high_20_aligned[i] and 
                adx_aligned[i] > 25 and 
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low with strong weekly downtrend
            elif (close[i] < donch_low_20_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  close[i] < ema21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below daily Donchian low OR trend weakens
            if (close[i] < donch_low_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above daily Donchian high OR trend weakens
            if (close[i] > donch_high_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyADX21_EMATrend"
timeframe = "12h"
leverage = 1.0