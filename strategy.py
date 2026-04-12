#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Chop_v2
Hypothesis: Daily KAMA trend with RSI momentum and weekly Choppiness filter.
KAMA adapts to market noise, reducing false signals in chop. RSI confirms momentum.
Chop filter avoids trend-following in ranging markets. Designed for low trade frequency
(7-25 trades/year) to minimize fee drag while capturing major trends in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Choppiness index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average) on daily closes
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility[10:]])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14) on daily closes
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on weekly data
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = df_1w['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1w['low'].rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero
    chop = np.where((hh - ll) != 0, chop, 50)
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA trend: price above/below KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # RSI momentum: oversold/overbought
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Choppiness filter: only trend-follow when market is trending (Chop < 38.2)
        trending_market = chop_aligned[i] < 38.2
        ranging_market = chop_aligned[i] > 61.8
        
        # Entry conditions
        long_entry = above_kama and rsi_oversold and trending_market
        short_entry = below_kama and rsi_overbought and trending_market
        
        # Exit conditions: opposite signal or market becomes ranging
        long_exit = (below_kama or rsi[i] > 50 or ranging_market)
        short_exit = (above_kama or rsi[i] < 50 or ranging_market)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals