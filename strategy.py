#!/usr/bin/env python3
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
    
    # === 1d data (primary) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (HTF for trend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 1d KAMA (Efficiency Ratio) for trend direction ===
    # Calculate Efficiency Ratio (ER) over 10 periods
    change_1d = np.abs(close_1d - np.roll(close_1d, 10))
    change_1d[0:10] = 0  # First 10 values will be handled by min_periods
    sum_abs_diff = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        sum_abs_diff[i] = sum_abs_diff[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    er_1d = np.where(sum_abs_diff != 0, change_1d / sum_abs_diff, 0)
    # Smoothing constants
    sc = (er_1d * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === 1d RSI (14 period) for momentum ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1d ATR (14 period) for volatility and stop ===
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1w trend filter: price above/below weekly EMA21 ===
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    
    # Warmup: enough for KAMA, RSI, ATR, and weekly EMA
    warmup = 50
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price below KAMA or 2x ATR stop
            if price < kama_val or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price above KAMA or 2x ATR stop
            if price > kama_val or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price above KAMA, RSI > 50, and above weekly EMA
            if price > kama_val and rsi_val > 50 and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: price below KAMA, RSI < 50, and below weekly EMA
            elif price < kama_val and rsi_val < 50 and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0