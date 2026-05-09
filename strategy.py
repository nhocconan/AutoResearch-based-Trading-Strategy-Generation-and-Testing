#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # KAMA calculation (daily)
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # RSI(14) calculation
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppy Index (14-period)
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - close_s.shift()), np.abs(low - close_s.shift()))))
    tr_sum = atr.rolling(window=14, min_periods=14).sum()
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # Weekly trend: EMA50 on weekly close
    weekly_close_s = pd.Series(df_1w['close'].values)
    ema50_1w = weekly_close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Enough for KAMA, RSI, Chop
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema50_1w_val = ema50_1w_aligned[i]
        
        if position == 0:
            # Enter long: Price above KAMA + RSI > 50 + Chop < 61.8 (trending) + weekly uptrend
            if close[i] > kama_val and rsi_val > 50 and chop_val < 61.8 and close[i] > ema50_1w_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below KAMA + RSI < 50 + Chop < 61.8 (trending) + weekly downtrend
            elif close[i] < kama_val and rsi_val < 50 and chop_val < 61.8 and close[i] < ema50_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below KAMA or RSI < 40
            if close[i] < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above KAMA or RSI > 60
            if close[i] > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals