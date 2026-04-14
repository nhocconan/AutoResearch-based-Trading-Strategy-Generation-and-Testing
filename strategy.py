#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA direction with RSI and weekly Chop filter
# Uses 1-day KAMA to determine trend direction, RSI(14) for overbought/oversold
# and weekly Chop to identify ranging markets. Only enters when KAMA direction
# aligns with RSI signal in trending markets (Chop < 38.2) or reverses in
# ranging markets (Chop > 61.8). Weekly Chop avoids whipsaws in low volatility.
# Position size: 0.25 (25% of capital)
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE for Chop calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Chop (61.8/38.2 levels)
    atr_period = 14
    high_low = df_1w['high'] - df_1w['low']
    high_close = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    low_close = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    max_high = df_1w['high'].rolling(window=14, min_periods=14).max()
    min_low = df_1w['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / 
                          np.log10(max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_values)
    
    # Calculate daily KAMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean()
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    # Start after sufficient data
    start = 50
    
    for i in range(start, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
            
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Trending market (Chop < 38.2): follow KAMA direction with RSI filter
            if chop_val < 38.2:
                if price > kama_val and rsi_val < 70:  # Uptrend, not overbought
                    position = 1
                    signals[i] = position_size
                elif price < kama_val and rsi_val > 30:  # Downtrend, not oversold
                    position = -1
                    signals[i] = -position_size
            # Ranging market (Chop > 61.8): mean reversion at RSI extremes
            elif chop_val > 61.8:
                if rsi_val < 30 and price > kama_val:  # Oversold but above KAMA
                    position = 1
                    signals[i] = position_size
                elif rsi_val > 70 and price < kama_val:  # Overbought but below KAMA
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI overbought in ranging market
            if price < kama_val or (chop_val > 61.8 and rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI oversold in ranging market
            if price > kama_val or (chop_val > 61.8 and rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0