#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR for Choppiness Index
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_h_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_l_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_h_14 - min_l_14
    
    # Choppiness Index: 100 * log10(sum_tr_14 / range_14) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    
    # Calculate ADX components
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr_14_smooth = pd.Series(atr_14).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14_smooth * 100)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14_smooth * 100)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12-period RSI for entry signal
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=12, min_periods=12).mean().values
    avg_loss = pd.Series(loss).rolling(window=12, min_periods=12).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 12, rsi])
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Chop regime filter: range-bound market (Chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # ADX filter: weak trend (ADX < 25)
        adx_filter = adx_aligned[i] < 25
        
        # RSI mean reversion: oversold/overbought with chop
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        long_entry = chop_filter and adx_filter and rsi_oversold
        short_entry = chop_filter and adx_filter and rsi_overbought
        
        # Exit when RSI returns to neutral zone
        long_exit = rsi[i] >= 50
        short_exit = rsi[i] <= 50
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Chop_ADX_RSI_MeanReversion"
timeframe = "12h"
leverage = 1.0