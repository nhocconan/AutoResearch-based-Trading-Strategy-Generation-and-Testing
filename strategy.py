#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 12h SMA for trend
    sma = pd.Series(prices['close']).rolling(window=20, min_periods=20).mean().values
    
    # 12h Bollinger Bands for mean reversion signals
    bb_std = pd.Series(prices['close']).rolling(window=20, min_periods=20).std(ddof=0).values
    upper_band = sma + 2.0 * bb_std
    lower_band = sma - 2.0 * bb_std
    
    # Daily ATR for volatility regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Weekly ATR for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_ma = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly ATR and its MA to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_1w_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(sma[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_1w_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        atr_1d = atr_1d_aligned[i]
        atr_1w = atr_1w_aligned[i]
        atr_1w_ma = atr_1w_ma_aligned[i]
        
        # Trend filter: only trade when weekly ATR is above its MA (trending market)
        trending = atr_1w > atr_1w_ma
        
        if position == 0 and trending:
            # Mean reversion entries at Bollinger Bands
            if price <= lower_band[i]:
                signals[i] = 0.25
                position = 1
            elif price >= upper_band[i]:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to SMA (mean reversion completion)
            if (position == 1 and price >= sma[i]) or (position == -1 and price <= sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_BollingerBandsMeanReversion_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0