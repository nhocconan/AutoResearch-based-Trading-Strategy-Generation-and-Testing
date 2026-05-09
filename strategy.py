#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_EMA_RSI_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 4h close (ER=10, fast=2, slow=30)
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate EMA21 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Calculate RSI14 on 1d close
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14_1d = 100 - (100 / (1 + rs))
    rsi14_1d = rsi14_1d.fillna(50).values
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Need enough data for KAMA, EMA21, RSI14
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(kama[i]) or 
            np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(rsi14_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        ema21_1d_val = ema21_1d_aligned[i]
        rsi14_1d_val = rsi14_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price > KAMA + 1d uptrend (EMA21) + RSI > 50 + volume spike
            if close[i] > kama_val and close[i] > ema21_1d_val and rsi14_1d_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Price < KAMA + 1d downtrend (EMA21) + RSI < 50 + volume spike
            elif close[i] < kama_val and close[i] < ema21_1d_val and rsi14_1d_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price < KAMA or 1d trend turns down (EMA21)
            if close[i] < kama_val or close[i] < ema21_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price > KAMA or 1d trend turns up (EMA21)
            if close[i] > kama_val or close[i] > ema21_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals