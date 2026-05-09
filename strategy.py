#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA calculation
    close_s = pd.Series(close)
    change = close_s.diff().abs()
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI calculation
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr14 / (hh14 - ll14)) / np.log10(14)
    
    # Weekly trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Align KAMA, RSI, Chop to daily
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi.values)
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(weekly_ema_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA bullish, RSI > 50, Chop < 61.8 (trending), price > weekly EMA
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 61.8 and 
                close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish, RSI < 50, Chop < 61.8, price < weekly EMA
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8 and 
                  close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA bearish OR Chop > 61.8 (ranging)
            if (close[i] < kama_aligned[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA bullish OR Chop > 61.8
            if (close[i] > kama_aligned[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals