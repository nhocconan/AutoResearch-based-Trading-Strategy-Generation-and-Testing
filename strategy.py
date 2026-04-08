#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation
    close_s = pd.Series(close)
    change = close_s.diff(10).abs()
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14)
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr2[0] = atr3[0] = 0
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)
    
    # Weekly trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA or RSI > 70
            if close[i] < kama[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above KAMA or RSI < 30
            if close[i] > kama[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs weekly EMA
            uptrend = close[i] > weekly_ema_aligned[i]
            downtrend = close[i] < weekly_ema_aligned[i]
            
            # Long: price above KAMA + RSI < 50 + chop > 61.8 (range) + uptrend
            if (close[i] > kama[i] and 
                rsi[i] < 50 and 
                chop[i] > 61.8 and
                uptrend):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA + RSI > 50 + chop > 61.8 (range) + downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] > 50 and 
                  chop[i] > 61.8 and
                  downtrend):
                position = -1
                signals[i] = -0.25
    
    return signals