#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI(14) + Choppiness regime filter
# Long when KAMA > previous KAMA (trend up) and RSI < 70 and Choppiness > 61.8 (range)
# Short when KAMA < previous KAMA (trend down) and RSI > 30 and Choppiness > 61.8
# Exit when KAMA direction changes or RSI reaches opposite extreme
# Uses KAMA for trend, RSI for momentum, Choppiness for regime filter to avoid whipsaw
# Target: 10-25 trades/year per symbol to stay within frequency limits for 1d
name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(price, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.sum(np.abs(np.diff(price, prepend=price[0])), axis=0) if len(price.shape) > 1 else np.sum(np.abs(np.diff(price, prepend=price[0])))
        # For 1D array
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.sum(np.abs(np.diff(price, prepend=price[0])))
        # Actually need to compute per element
        er = np.zeros_like(price)
        for i in range(er_length, len(price)):
            if np.sum(np.abs(np.diff(price[i-er_length:i+1]))) > 0:
                er[i] = np.abs(price[i] - price[i-er_length]) / np.sum(np.abs(np.diff(price[i-er_length:i+1])))
            else:
                er[i] = 0
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # Simpler: use EMA as proxy for trend (but we'll implement proper KAMA)
    # Actually, let's use a simpler adaptive approach
    er_length = 10
    fast_sc = 2
    slow_sc = 30
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(er_length, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_length:i+1])))
    
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    rsi_length = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[rsi_length] = np.mean(gain[1:rsi_length+1])
    avg_loss[rsi_length] = np.mean(loss[1:rsi_length+1])
    
    for i in range(rsi_length+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_length-1) + gain[i]) / rsi_length
        avg_loss[i] = (avg_loss[i-1] * (rsi_length-1) + loss[i]) / rsi_length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index
    chop_length = 14
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr
    
    # Sum of ATR over chop_length periods
    sum_atr = np.zeros_like(close)
    for i in range(chop_length, len(close)):
        sum_atr[i] = np.sum(atr[i-chop_length+1:i+1])
    
    # Highest high and lowest low over chop_length periods
    highest_high = np.zeros_like(close)
    lowest_low = np.zeros_like(close)
    for i in range(chop_length-1, len(close)):
        highest_high[i] = np.max(high[i-chop_length+1:i+1])
        lowest_low[i] = np.min(low[i-chop_length+1:i+1])
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.zeros_like(close)
    for i in range(chop_length-1, len(close)):
        if sum_atr[i] > 0 and range_hl[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(chop_length)
        else:
            chop[i] = 50  # neutral
    
    # Get weekly data for additional filter (optional)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) > 0:
        weekly_close = df_1w['close'].values
        # Simple weekly trend: price above weekly EMA21
        weekly_ema = np.zeros_like(weekly_close)
        weekly_ema[0] = weekly_close[0]
        for i in range(1, len(weekly_close)):
            weekly_ema[i] = 0.1 * weekly_close[i] + 0.9 * weekly_ema[i-1]
        # Align to daily
        weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    else:
        weekly_ema_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_length, chop_length, er_length) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            (i > 0 and np.isnan(kama[i-1]))):
            signals[i] = 0.0
            continue
        
        # Optional: weekly filter (only trade in direction of weekly trend)
        weekly_ok = True
        if not np.isnan(weekly_ema_aligned[i]):
            weekly_ok = close[i] > weekly_ema_aligned[i]  # only long above weekly EMA
        
        if position == 0:
            # Long entry: KAMA rising (trend up), RSI not overbought, choppy market
            if kama[i] > kama[i-1] and rsi[i] < 70 and chop[i] > 61.8 and weekly_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA falling (trend down), RSI not oversold, choppy market
            elif kama[i] < kama[i-1] and rsi[i] > 30 and chop[i] > 61.8 and not weekly_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down OR RSI overbought
            if kama[i] < kama[i-1] or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up OR RSI oversold
            if kama[i] > kama[i-1] or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals