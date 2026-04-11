#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: 1d KAMA trend with RSI momentum and weekly chop regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets. RSI filters for momentum strength, avoiding weak trends. Weekly chop regime ensures we only trade in trending markets (low chop) or mean-revert in ranging markets (high chop), adapting to changing market conditions. This approach reduces whipsaws and improves trend capture.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1d KAMA(14, 2, 30) for trend
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.subtract(close[1:11], close[:-11]), axis=0) if len(close) > 11 else np.abs(np.diff(close)))
    # Simplified ER calculation using pandas
    close_series = pd.Series(close)
    change_abs = close_series.diff(10).abs()
    volatility_sum = close_series.diff().abs().rolling(window=10, min_periods=1).sum()
    er = change_abs / volatility_sum.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama = kama  # already aligned to 1d
    
    # 1d RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly chop regime: choppy if >61.8, trending if <38.2
    # Chop = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    atr_1w = []
    tr_1w = []
    for i in range(len(df_1w)):
        if i == 0:
            tr = df_1w['high'].iloc[i] - df_1w['low'].iloc[i]
        else:
            tr = max(
                df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
                abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
                abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
            )
        tr_1w.append(tr)
        atr_1w.append(np.mean(tr_1w[-14:]) if len(tr_1w) >= 14 else np.nan)
    
    chop_raw = []
    for i in range(len(df_1w)):
        if i < 13:
            chop_raw.append(np.nan)
        else:
            sum_atr = sum(tr_1w[i-13:i+1])
            highest_high = max(df_1w['high'].iloc[i-13:i+1])
            lowest_low = min(df_1w['low'].iloc[i-13:i+1])
            if highest_high == lowest_low:
                chop_raw.append(100)
            else:
                chop_raw.append(100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14))
    
    chop = np.array(chop_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime-based logic
        if chop_aligned[i] < 38.2:  # Trending market
            # Trend following: KAMA direction + RSI momentum
            if close[i] > kama[i] and rsi[i] > 50 and position != 1:
                position = 1
                signals[i] = 0.25
            elif close[i] < kama[i] and rsi[i] < 50 and position != -1:
                position = -1
                signals[i] = -0.25
        else:  # Choppy/ranging market
            # Mean reversion: fade extreme RSI
            if rsi[i] < 30 and position != 1:  # Oversold -> long
                position = 1
                signals[i] = 0.25
            elif rsi[i] > 70 and position != -1:  # Overbought -> short
                position = -1
                signals[i] = -0.25
        
        # Exit conditions
        if position == 1:
            if chop_aligned[i] < 38.2 and (close[i] < kama[i] or rsi[i] < 50):  # Trend end
                position = 0
                signals[i] = 0.0
            elif chop_aligned[i] >= 38.2 and rsi[i] > 50:  # RSI no longer oversold
                position = 0
                signals[i] = 0.0
        elif position == -1:
            if chop_aligned[i] < 38.2 and (close[i] > kama[i] or rsi[i] > 50):  # Trend end
                position = 0
                signals[i] = 0.0
            elif chop_aligned[i] >= 38.2 and rsi[i] < 50:  # RSI no longer overbought
                position = 0
                signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals