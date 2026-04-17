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
    
    # Get daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility and stop
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get weekly data for regime filter (choppiness)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly True Range and ADX-like calculation for chop filter
    high_low_w = high_1w - low_1w
    high_close_w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(high_low_w, np.maximum(high_close_w, low_close_w))
    tr_w[0] = high_low_w[0]
    
    # Calculate ATR and price range for choppiness
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    max_high_w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_w = max_high_w - min_low_w
    
    # Choppiness Index: higher = more ranging, lower = more trending
    chop_w = 100 * np.log10(atr_w * 14 / range_w) / np.log10(14)
    chop_w_aligned = align_htf_to_ltf(prices, df_1w, chop_w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(chop_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA34, chop indicates trending (>50), and momentum up
            if close[i] > ema34_1d_aligned[i] and chop_w_aligned[i] > 50 and close[i] > close[max(0, i-1)]:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA34, chop indicates trending (>50), and momentum down
            elif close[i] < ema34_1d_aligned[i] and chop_w_aligned[i] > 50 and close[i] < close[max(0, i-1)]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34 OR chop indicates ranging (<40)
            if close[i] < ema34_1d_aligned[i] or chop_w_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34 OR chop indicates ranging (<40)
            if close[i] > ema34_1d_aligned[i] or chop_w_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA34_ChopFilter_Trend"
timeframe = "4h"
leverage = 1.0