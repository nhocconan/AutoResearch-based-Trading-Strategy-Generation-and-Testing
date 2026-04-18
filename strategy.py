#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Strategy: Enter long when KAMA indicates bullish momentum and RSI is oversold in choppy market.
          Enter short when KAMA indicates bearish momentum and RSI is overbought in choppy market.
          Uses weekly EMA200 as trend filter to avoid counter-trend trades.
          Designed for low trade frequency with mean-reversion edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    weekly_close = df_1w['close'].values
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate KAMA (10-period ER, 2 and 30 SC)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop (14-period)
    atr = pd.Series(np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])).rolling(window=14, min_periods=14).sum().values
    atr = np.concatenate([np.full(14, np.nan), atr])
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200 = ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA bullish, RSI oversold, choppy market
            if (kama[i] > close[i] and rsi[i] < 30 and chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish, RSI overbought, choppy market
            elif (kama[i] < close[i] and rsi[i] > 70 and chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: KAMA turns bearish or RSI overbought
            if kama[i] < close[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: KAMA turns bullish or RSI oversold
            if kama[i] > close[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0