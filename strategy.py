#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Align weekly data to daily bars
    pivots_high = align_htf_to_ltf(prices, df_1w, high_1w)
    pivots_low = align_htf_to_ltf(prices, df_1w, low_1w)
    pivots_close = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate Camarilla H3 and L3 levels (entry levels)
    H3 = pivots_close + (pivots_high - pivots_low) * 1.1 / 4
    L3 = pivots_close - (pivots_high - pivots_low) * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    strong_volume = volume > (vol_ma * 1.5)
    
    # Chop index on daily for regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(pivots_close[i]) or
            np.isnan(chop[i]) or np.isnan(strong_volume[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime: Chop < 40 = trending (favor breakouts), Chop > 60 = ranging (avoid)
        trending_regime = chop[i] < 40
        
        # Long: price breaks above H3 in trending market with volume
        long_signal = (close[i] > H3[i] and trending_regime and strong_volume[i])
        
        # Short: price breaks below L3 in trending market with volume
        short_signal = (close[i] < L3[i] and trending_regime and strong_volume[i])
        
        # Exit: chop increases (range) or price returns to pivot
        exit_long = (position == 1 and (chop[i] > 60 or close[i] < pivots_close[i]))
        exit_short = (position == -1 and (chop[i] > 60 or close[i] > pivots_close[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals