#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_sc = 0.666  # EMA constant for fast EMA (2)
    slow_sc = 0.0645 # EMA constant for slow EMA (30)
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period volatility
    # Fix dimensions: change has length n-10, vol has length n-1
    # We'll compute ER for index i using change[i:i+10] and vol[i:i+10]
    er = np.full(n, np.nan)
    for i in range(10, n):
        if vol[i-10:i] > 0:  # vol[i-10:i] corresponds to periods i-10 to i-1
            er[i] = change[i-10:i].sum() / vol[i-10:i].sum()
        else:
            er[i] = 0
    
    # Calculate SC: [ER * (fast_sc - slow_sc) + slow_sc]^2
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # First average
    avg_gain[13] = gain[1:14].mean()
    avg_loss[13] = loss[1:14].mean()
    
    # Wilder smoothing
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Chopiness Index (14-period) - range detection
    atr = np.full(n, np.nan)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # ATR calculation
    atr[13] = tr[1:14].mean()
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Chop calculation: 100 * log10(sum(ATR,14) / (HHH - LLL)) / log10(14)
    sum_atr14 = np.full(n, np.nan)
    hh_l = np.full(n, np.nan)
    ll_h = np.full(n, np.nan)
    
    for i in range(13, n):
        sum_atr14[i] = tr[i-13:i+1].sum()  # 14-period sum of TR
        hh_l[i] = high[i-13:i+1].max() - low[i-13:i+1].min()
    
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if hh_l[i] > 0:
            chop[i] = 100 * np.log10(sum_atr14[i] / hh_l[i]) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when no range
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA, RSI > 50, low chop (trending)
            if close[i] > kama[i] and rsi[i] > 50 and chop[i] < 38.2 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, RSI < 50, low chop (trending)
            elif close[i] < kama[i] and rsi[i] < 50 and chop[i] < 38.2 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI < 40 OR high chop (ranging)
            if close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI > 60 OR high chop (ranging)
            if close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals