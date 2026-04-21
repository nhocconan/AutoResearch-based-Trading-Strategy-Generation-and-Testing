#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly ATR for volatility filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                  np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === Daily ADX for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h ATR for breakout threshold ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    tr_6h = np.maximum(high_6h[1:] - low_6h[1:], 
                       np.maximum(np.abs(high_6h[1:] - close_6h[:-1]), 
                                  np.abs(low_6h[1:] - close_6h[:-1])))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=10, min_periods=10).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        atr_1w_val = atr_1w_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_6h_val = atr_6h[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: strong weekly volatility + strong daily trend + bullish 6h bar + volume
            if (atr_1w_val > 0 and  # volatility filter
                adx_val > 25 and    # strong trend
                price_close > price_open and  # bullish bar
                (price_close - price_open) > 0.5 * atr_6h_val and  # significant move
                vol_ratio_val > 1.3):  # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: strong weekly volatility + strong daily trend + bearish 6h bar + volume
            elif (atr_1w_val > 0 and    # volatility filter
                  adx_val > 25 and      # strong trend
                  price_close < price_open and  # bearish bar
                  (price_open - price_close) > 0.5 * atr_6h_val and  # significant move
                  vol_ratio_val > 1.3):  # volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility drops or trend weakens
            if (atr_1w_val < 0.8 * atr_1w_aligned[i-1] and i > 30) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyVolatility_Trend_Momentum"
timeframe = "6h"
leverage = 1.0