#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + chop filter with 1w trend filter
# KAMA adapts to market noise, reducing whipsaw in choppy markets
# RSI(14) > 50 for long, < 50 for short ensures momentum alignment
# Choppiness index > 61.8 avoids trending markets (use mean reversion only in chop)
# 1w EMA13 filter ensures alignment with higher timeframe trend
# Designed for 1d timeframe with selective entries to avoid overtrading
# Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 13-period EMA on 1w timeframe for trend filter
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Calculate KAMA on daily closes
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle erosion correctly
    er = np.zeros_like(change)
    for i in range(er_length, len(close)):
        if volatility[i-er_length:i] > 0:
            er[i] = change[i-er_length] / volatility[i-er_length:i].sum()
        else:
            er[i] = 0
    # Pad beginning with zeros
    er = np.concatenate([np.zeros(er_length), er])
    
    # Calculate smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50
    
    # Calculate Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])  # Align with close
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if hh[i] - ll[i] != 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50
    chop[:14] = 50
    
    # Align indicators to daily timeframe
    kama_aligned = kama  # Already on daily
    rsi_aligned = rsi    # Already on daily
    chop_aligned = chop  # Already on daily
    ema13_1w_aligned = ema13_1w_aligned  # Already aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(chop_aligned[i]) or np.isnan(ema13_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        chop_high = chop_aligned[i] > 61.8  # Choppy market
        chop_low = chop_aligned[i] < 38.2   # Trending market
        
        # 1w trend filter
        is_uptrend_1w = close[i] > ema13_1w_aligned[i]
        is_downtrend_1w = close[i] < ema13_1w_aligned[i]
        
        if position == 0:
            # Long entry: price > KAMA + RSI > 50 + chop > 61.8 + 1w uptrend
            long_signal = (price_above_kama and rsi_bullish and chop_high and is_uptrend_1w)
            
            # Short entry: price < KAMA + RSI < 50 + chop > 61.8 + 1w downtrend
            short_signal = (price_below_kama and rsi_bearish and chop_high and is_downtrend_1w)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA OR RSI < 40
            if price_below_kama or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA OR RSI > 60
            if price_above_kama or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_1wTrendFilter"
timeframe = "1d"
leverage = 1.0