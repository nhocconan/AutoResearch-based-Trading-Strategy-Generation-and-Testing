#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA parameters
    fast = 2
    slow = 30
    lookback = 10
    
    # Calculate ER (Efficiency Ratio) for KAMA
    change = np.abs(np.diff(close, n=lookback))
    vol = np.sum(np.abs(np.diff(close)), axis=0) if lookback == 1 else np.array([
        np.sum(np.abs(np.diff(close[i:i+lookback]))) if i+lookback <= len(close) else np.nan
        for i in range(len(close))
    ])
    # Simplified ER calculation using pandas for efficiency
    close_series = pd.Series(close)
    change = close_series.diff(lookback).abs()
    vol = close_series.diff().abs().rolling(lookback).sum()
    er = change / vol.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    sc = np.nan_to_num(sc, nan=0.0)
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14)
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr[0] = 0
    
    # True range for chop calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    
    # Sum of ATR and TR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr / sum_atr) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging
        chop_trending = chop[i] < 38.2
        chop_ranging = chop[i] > 61.8
        
        # Entry conditions
        # Long: Price > KAMA AND RSI > 50 AND trending regime AND volume confirmation
        if close[i] > kama_aligned[i] and rsi[i] > 50 and chop_trending and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price < KAMA AND RSI < 50 AND trending regime AND volume confirmation
        elif close[i] < kama_aligned[i] and rsi[i] < 50 and chop_trending and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite signal or ranging market
        elif position == 1 and (close[i] < kama_aligned[i] or chop_ranging):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_aligned[i] or chop_ranging):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals