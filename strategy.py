#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_KAMA_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w: KAMA for trend direction ===
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, k=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.full_like(close_1w, np.nan, dtype=float)
    kama[9] = close_1w[9]  # seed
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i-10] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # === 1d: RSI for overbought/oversold ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # === 1d: Choppiness Index for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    # Align 1w KAMA
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    # Align 1d RSI and Chop
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        kama = kama_1w_aligned[i]
        rsi = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama) or np.isnan(rsi) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.3x 30-period average
        if i >= 30:
            vol_ma = np.mean(volume[i-30:i])
            vol_condition = current_volume > 1.3 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long: price > weekly KAMA (uptrend), RSI < 30 (oversold), chop > 50 (rangy/mild trend)
            if current_close > kama and rsi < 30 and chop_val > 50 and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price < weekly KAMA (downtrend), RSI > 70 (overbought), chop > 50
            elif current_close < kama and rsi > 70 and chop_val > 50 and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Exit long: price crosses below weekly KAMA OR RSI > 70 (overbought)
            if current_close <= kama or rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly KAMA OR RSI < 30 (oversold)
            if current_close >= kama or rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals