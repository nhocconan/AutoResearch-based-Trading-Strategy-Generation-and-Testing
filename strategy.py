#!/usr/bin/env python3
"""
12h_1d_kama_rsi_chop_v1
Strategy: 12h KAMA direction + RSI + chop filter with volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: KAMA identifies trend direction, RSI filters overbought/oversold, Choppiness Index avoids ranging markets, and volume confirms momentum. Designed for low trade frequency (<25/year) with high win rate in both bull and bear markets by avoiding false signals in chop and catching strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA (12h) - Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    # Use rolling sum for efficiency ratio calculation
    change_sum = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.divide(change_sum, volatility_sum, out=np.zeros_like(change_sum), where=volatility_sum!=0)
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) - uses 1d data for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = []
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True range for chop calculation
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (hh_14 - ll_14)) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation (20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend direction: price relative to KAMA
        above_kama = price_close > kama[i]
        below_kama = price_close < kama[i]
        
        # RSI filters: avoid extreme overbought/oversold
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Chop filter: only trade when NOT choppy (trending market)
        trending = chop_aligned[i] < 50  # Below 50 = trending, above = chopping
        
        # Volume confirmation
        vol_confirmed = vol_surge[i]
        
        # Long: price above KAMA, RSI not overbought, trending, volume surge
        long_signal = above_kama and rsi_not_overbought and trending and vol_confirmed
        
        # Short: price below KAMA, RSI not oversold, trending, volume surge
        short_signal = below_kama and rsi_not_oversold and trending and vol_confirmed
        
        # Exit when price crosses back through KAMA or RSI reaches extreme
        exit_long = position == 1 and (price_close < kama[i] or rsi[i] > 75)
        exit_short = position == -1 and (price_close > kama[i] or rsi[i] < 25)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals