#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Trend_Filter_With_RSI_And_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA, RSI, and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA calculation (10-period ER, 2/30 SC)
    close_series = pd.Series(df_1d['close'])
    change = abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [np.nan] * len(close_series)
    if len(close_series) > 0:
        kama[0] = close_series.iloc[0]
        for i in range(1, len(close_series)):
            if not np.isnan(sc.iloc[i]):
                kama[i] = kama[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    kama = np.array(kama)
    
    # RSI (14-period)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Choppiness Index (14-period)
    atr = np.maximum(high - low, np.maximum(abs(high - close_series.shift(1)), abs(low - close_series.shift(1))))
    tr_sum = pd.Series(atr).rolling(14, min_periods=14).sum()
    hh = high.rolling(14, min_periods=14).max()
    ll = low.rolling(14, min_periods=14).min()
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = chop.values
    
    # Align to 12h
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]) or np.isnan(chop_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_12h[i]
        rsi_val = rsi_12h[i]
        chop_val = chop_12h[i]
        
        if position == 0:
            # Enter long: price > KAMA, RSI > 50, chop < 61.8 (trending)
            if close[i] > kama_val and rsi_val > 50 and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA, RSI < 50, chop < 61.8 (trending)
            elif close[i] < kama_val and rsi_val < 50 and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI < 45 or chop > 61.8 (choppy)
            if close[i] < kama_val or rsi_val < 45 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI > 55 or chop > 61.8 (choppy)
            if close[i] > kama_val or rsi_val > 55 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals