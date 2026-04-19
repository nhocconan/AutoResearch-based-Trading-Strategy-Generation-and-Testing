#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_ChopFilter_V1"
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
    
    # Get 1d data for calculations (already daily)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on 1d close (ER=10, fast=2, slow=30)
    close_s = pd.Series(close_1d)
    change = abs(close_s.diff(10))
    volatility = abs(close_s.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily (no shift needed as 1d data)
    kama_aligned = kama
    
    # Calculate RSI(14) on 1d close
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi_aligned = rsi.values
    
    # Calculate EMA34 on weekly close for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Choppiness Index on 1d (period=14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(14, n):
        if atr[i] > 0 and highest_high[i] > lowest_low[i]:
            sum_atr = pd.Series(tr).iloc[i-13:i+1].sum()
            chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    chop[:14] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
            
        # Chop filter: only trade when market is trending (CHOP < 38.2) or extreme reversal (CHOP > 61.8)
        trending_market = chop[i] < 38.2
        extreme_chop = chop[i] > 61.8
        
        if position == 0:
            # Long when price > KAMA, RSI > 50, and (trending market OR extreme chop reversal)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and (trending_market or extreme_chop):
                signals[i] = 0.25
                position = 1
            # Short when price < KAMA, RSI < 50, and (trending market OR extreme chop reversal)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and (trending_market or extreme_chop):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price < KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price > KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals