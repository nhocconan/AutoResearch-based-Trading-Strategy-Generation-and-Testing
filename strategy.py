#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_KAMA_Trend_Volume_Confirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constants for KAMA (10-period)
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # Initialize
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align daily KAMA to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily ADX for trend strength (14-period)
    # Calculate True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    
    # Calculate Directional Movement
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM
    tr_period = 14
    atr = np.full_like(tr1, np.nan)
    for i in range(tr_period, len(tr1)):
        if i == tr_period:
            atr[i] = np.nansum(tr1[i-tr_period+1:i+1])
        else:
            atr[i] = atr[i-1] - (atr[i-1] / tr_period) + tr1[i]
    
    plus_di = 100 * np.full_like(up_move, np.nan)
    minus_di = 100 * np.full_like(down_move, np.nan)
    for i in range(tr_period, len(up_move)):
        if atr[i+1] != 0:
            plus_di[i] = 100 * (np.nansum(plus_dm[i-tr_period+1:i+1]) / atr[i+1])
            minus_di[i] = 100 * (np.nansum(minus_dm[i-tr_period+1:i+1]) / atr[i+1])
    
    dx = np.full_like(plus_di, np.nan)
    for i in range(len(dx)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.full_like(dx, np.nan)
    for i in range(14, len(dx)):
        if i == 14:
            adx[i] = np.nanmean(dx[1:15])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama = kama_aligned[i]
        adx = adx_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        strong_trend = adx > 25
        
        if position == 0:
            # Long: price above KAMA with volume and strong trend
            if price > kama and volume_confirmed and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume and strong trend
            elif price < kama and volume_confirmed and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below KAMA or weak trend
            if price < kama or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above KAMA or weak trend
            if price > kama or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals