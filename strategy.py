#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On 1d timeframe, use KAMA (adaptive trend) for direction, RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering. Enter long when KAMA is rising, RSI > 50, and CHOP < 38.2 (trending regime). Enter short when KAMA is falling, RSI < 50, and CHOP < 38.2. Uses discrete position size 0.25. Designed for 15-25 trades/year on 1d by requiring alignment of trend, momentum, and regime filters, reducing overtrading while capturing sustained moves in both bull and bear markets. Weekly trend filter (close > weekly EMA50) ensures alignment with higher timeframe structure.
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
    
    # Get 1d data for KAMA, RSI, Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators ===
    # KAMA (adaptive trend) - using close prices
    close_1d = pd.Series(df_1d['close'].values)
    # Efficiency Ratio
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    kama_values = kama
    
    # RSI(14)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Choppiness Index(14)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - close_1d.shift(1))), np.abs(low - close_1d.shift(1)))).rolling(14).mean()
    max_high = close_1d.rolling(14).max()
    min_low = close_1d.rolling(14).min()
    chop = 100 * np.log10(atr_14.rolling(14).sum() / np.log10(max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    # === 1w Trend Filter ===
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to lower timeframe (prices)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1w EMA warmup, 1d indicator warmup
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d conditions
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        chop_low = chop_aligned[i] < 38.2  # trending regime
        
        # 1w trend filter
        trend_1w_uptrend = close[i] > ema_50_1w[i]
        trend_1w_downtrend = close[i] < ema_50_1w[i]
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, CHOP < 38.2 (trending), 1w uptrend
            long_signal = kama_rising and rsi_above_50 and chop_low and trend_1w_uptrend
            
            # Short: KAMA falling, RSI < 50, CHOP < 38.2 (trending), 1w downtrend
            short_signal = kama_falling and rsi_below_50 and chop_low and trend_1w_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA falling OR RSI < 50 OR CHOP > 61.8 (ranging) OR 1w trend turns down
            if (kama_falling or not rsi_above_50 or chop_aligned[i] > 61.8 or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rising OR RSI > 50 OR CHOP > 61.8 (ranging) OR 1w trend turns up
            if (kama_rising or not rsi_below_50 or chop_aligned[i] > 61.8 or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0