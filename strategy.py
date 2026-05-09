#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Direction_RSI_Filter_Chop_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 12h data
    close_series = pd.Series(close)
    # Efficiency ratio
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Chop(14)
    atr_1d = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).mean()
    highest_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    chop_denom = highest_1d - lowest_1d
    chop_denom = chop_denom.replace(0, np.nan)
    chop = 100 * np.log10(atr_1d.rolling(window=14, min_periods=14).sum() / chop_denom) / np.log10(14)
    chop = chop.fillna(50).values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI > 50, Chop < 61.8 (trending)
            if close[i] > kama[i] and rsi_1d_aligned[i] > 50 and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 50, Chop < 61.8 (trending)
            elif close[i] < kama[i] and rsi_1d_aligned[i] < 50 and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below KAMA OR Chop > 61.8 (choppy)
            if close[i] < kama[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above KAMA OR Chop > 61.8 (choppy)
            if close[i] > kama[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals