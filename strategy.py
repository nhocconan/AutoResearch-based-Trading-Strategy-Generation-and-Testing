#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Three_Screen_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA200 for trend filter (weekly timeframe proxy)
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1w RSI(14) for overbought/oversold filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 4h Williams %R(14) for entry timing
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or \
           np.isnan(willr[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: 1d EMA200
        bullish_trend = price > ema200_1d_aligned[i]
        bearish_trend = price < ema200_1d_aligned[i]
        
        # 1w RSI filter: avoid extremes
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Entry conditions
        if position == 0:
            # Long: bullish trend, not overbought weekly, Williams %R oversold bounce
            if bullish_trend and rsi_not_overbought and willr[i] < -80 and willr[i] > willr[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: bearish trend, not oversold weekly, Williams %R overbought bounce
            elif bearish_trend and rsi_not_oversold and willr[i] > -20 and willr[i] < willr[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend reversal or Williams %R overbought
            if not bullish_trend or willr[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend reversal or Williams %R oversold
            if not bearish_trend or willr[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals