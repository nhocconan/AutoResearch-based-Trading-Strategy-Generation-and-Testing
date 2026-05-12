#3/23/2025
#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: On daily timeframe, use KAMA to determine trend direction and RSI for momentum.
# Enter long when KAMA is rising, RSI > 50, and Chop > 61.8 (range) for mean reversion.
# Enter short when KAMA is falling, RSI < 50, and Chop > 61.8 (range) for mean reversion.
# Exit when RSI crosses 50 or Chop < 38.2 (trending).
# Uses weekly trend filter: only trade in direction of weekly EMA34.
# Designed for low frequency (target 10-25 trades/year) to minimize fee drag.
# Works in both bull and bear markets by adapting to regime via Chop.

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Pad beginning with zeros
    er = np.concatenate([np.full(er_length, np.nan), er])
    
    # Calculate SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]  # seed
    for i in range(er_length + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop (Choppiness Index) - 14 period
    atr = np.full_like(close, np.nan)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = np.where(np.isnan(tr), 0, tr)  # simple fill for first element
    
    # Wilder's smoothing for ATR
    atr_smoothed = np.full_like(close, np.nan)
    atr_smoothed[14] = np.mean(atr[1:15])
    for i in range(15, n):
        atr_smoothed[i] = (atr_smoothed[i-1] * 13 + atr[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    # Chop calculation
    sum_atr14 = np.full_like(close, np.nan)
    for i in range(14, n):
        sum_atr14[i] = np.sum(atr_smoothed[i-13:i+1])
    
    chop = np.full_like(close, np.nan)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            log_val = np.log10(sum_atr14[i] / (highest_high[i] - lowest_low[i]))
            chop[i] = 100 * log_val / np.log10(14)
        else:
            chop[i] = 50  # avoid division by zero
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Align KAMA, RSI, Chop (they are already LTF, but ensure no NaN)
    kama_aligned = kama  # already calculated on LTF
    rsi_aligned = rsi
    chop_aligned = chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        kama_prev = kama_aligned[i-1]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        weekly_trend = ema34_1w_aligned[i]
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # LONG: KAMA rising, RSI > 50, Chop > 61.8 (range), and price > weekly EMA34
            if kama_rising and rsi_val > 50 and chop_val > 61.8 and close[i] > weekly_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, Chop > 61.8 (range), and price < weekly EMA34
            elif kama_falling and rsi_val < 50 and chop_val > 61.8 and close[i] < weekly_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI < 50 OR Chop < 38.2 (trending)
            if not kama_rising or rsi_val < 50 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI > 50 OR Chop < 38.2 (trending)
            if not kama_falling or rsi_val > 50 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals