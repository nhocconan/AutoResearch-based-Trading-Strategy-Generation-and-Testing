#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_DailyKAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 14 period
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14 period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14 period)
    atr = np.zeros_like(close_1d)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close_1d[0]), np.abs(low[0]-close_1d[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high - lowest_low)) / np.log10(14)
    
    # Align indicators to 1d timeframe
    kama_1d = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi)
    chop_1d = align_htf_to_ltf(prices, df_1d, chop)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]) or 
            np.isnan(ema_34_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Choppiness regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
        is_ranging = chop_1d[i] > 61.8
        is_trending = chop_1d[i] < 38.2
        
        if position == 0:
            # Long: price > KAMA, RSI < 40 (oversold), in ranging market
            if (close[i] > kama_1d[i] and 
                rsi_1d[i] < 40 and 
                is_ranging):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI > 60 (overbought), in ranging market
            elif (close[i] < kama_1d[i] and 
                  rsi_1d[i] > 60 and 
                  is_ranging):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI > 60
            if close[i] < kama_1d[i] or rsi_1d[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI < 40
            if close[i] > kama_1d[i] or rsi_1d[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals