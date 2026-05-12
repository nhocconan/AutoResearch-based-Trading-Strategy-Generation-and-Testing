#!/usr/bin/env python3
name = "1d_KAMA_20_RSI_14_Chop_14_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly KAMA (ER=10) for trend direction
    close_1w_series = pd.Series(close_1w)
    change = abs(close_1w_series.diff(10))
    volatility = close_1w_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_1w = [close_1w[0]]
    for i in range(1, len(close_1w)):
        kama_1w.append(kama_1w[-1] + sc.iloc[i] * (close_1w[i] - kama_1w[-1]))
    kama_1w = np.array(kama_1w)
    kama_1w_1d = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # === DAILY INDICATORS ===
    # KAMA(20) for entry timing
    close_series = pd.Series(close)
    change_daily = abs(close_series.diff(10))
    volatility_daily = close_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er_daily = change_daily / volatility_daily.replace(0, 1e-10)
    sc_daily = (er_daily * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_daily = [close[0]]
    for i in range(1, len(close)):
        kama_daily.append(kama_daily[-1] + sc_daily.iloc[i] * (close[i] - kama_daily[-1]))
    kama_daily = np.array(kama_daily)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Choppiness Index(14)
    atr1 = np.maximum(high - low, 
                      np.maximum(abs(high - np.roll(close, 1)), 
                                 abs(low - np.roll(close, 1))))
    atr1[0] = np.nan
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_1d[i]) or np.isnan(kama_daily[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Weekly KAMA (uptrend) + RSI < 30 (oversold) + Chop > 61.8 (range)
            if (close[i] > kama_1w_1d[i] and 
                rsi[i] < 30 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Weekly KAMA (downtrend) + RSI > 70 (overbought) + Chop > 61.8 (range)
            elif (close[i] < kama_1w_1d[i] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI > 50 (momentum shift) OR Chop < 38.2 (trending)
            if rsi[i] > 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (momentum shift) OR Chop < 38.2 (trending)
            if rsi[i] < 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals