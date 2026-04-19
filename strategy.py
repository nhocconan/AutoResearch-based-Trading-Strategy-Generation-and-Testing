#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and chop filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly
    # Proper ER calculation
    price_change = np.abs(np.diff(close_1d, k=10, prepend=close_1d[:10]))
    volatility_sum = np.convolve(np.abs(np.diff(close_1d)), np.ones(10), 'same')
    volatility_sum[:9] = np.cumsum(np.abs(np.diff(close_1d))[:9])[::-1]
    volatility_sum[-9:] = np.cumsum(np.abs(np.diff(close_1d))[-9:])
    er = np.where(volatility_sum != 0, price_change / volatility_sum, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period) on daily
    atr1 = np.abs(high_1d - low_1d)
    atr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    atr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) != 0, chop, 50)
    
    # Align daily values to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        trending = chop_val < 38.2  # Trending regime
        ranging = chop_val > 61.8   # Ranging regime
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, trending market, volume confirmation
            if price > kama_val and rsi_val > 50 and trending and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, trending market, volume confirmation
            elif price < kama_val and rsi_val < 50 and trending and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price < KAMA or RSI < 40 or chop > 61.8 (range)
            if price < kama_val or rsi_val < 40 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price > KAMA or RSI > 60 or chop > 61.8 (range)
            if price > kama_val or rsi_val > 60 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals