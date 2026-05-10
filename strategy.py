#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Chop
# Hypothesis: KAMA direction determines trend bias on daily, RSI provides entry timing
# within trend, and Choppiness Index filters for trending regimes. Works in bull
# (trend following) and bear (avoids chop, takes strong trend reversals).

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0

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
    
    # 1d KAMA trend
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = kama > np.roll(kama, 1)  # Today's KAMA > yesterday's
    
    # 1d RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 1d Choppiness Index(14)
    atr = np.abs(high - low)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators warm up
    
    for i in range(start_idx, n):
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(trend_1w_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50 (bullish momentum), chop < 61.8 (trending)
            if (kama_dir[i] and rsi[i] > 50 and chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50 (bearish momentum), chop < 61.8 (trending)
            elif (not kama_dir[i] and rsi[i] < 50 and chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA down or chop > 61.8 (choppy) or RSI < 40
            if (not kama_dir[i] or chop[i] > 61.8 or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA up or chop > 61.8 (choppy) or RSI > 60
            if (kama_dir[i] or chop[i] > 61.8 or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals