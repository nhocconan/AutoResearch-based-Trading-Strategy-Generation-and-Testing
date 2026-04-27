#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: 4h KAMA trend direction + RSI extreme + chop regime filter captures high-probability continuation moves in both bull and bear markets. KAMA adapts to volatility, reducing whipsaws. RSI < 30 or > 70 provides momentum confirmation. Chop > 61.8 avoids breakouts in ranging markets. Discrete sizing (0.30) limits fee drawdown. Target: 75-200 trades over 4 years.
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
    
    # Calculate KAMA (primary TF, 4h)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    change = np.concatenate([np.full(10, np.nan), change])  # align to original index
    volatility = np.abs(np.diff(close, n=1))  # |close[t] - close[t-1]|
    volatility = np.concatenate([np.array([np.nan]), volatility])  # align
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = change / vol_sum
    er = np.nan_to_num(er, nan=0.0)  # replace NaN with 0
    
    # Smoothing constants: fastest SC=2/(2+1)=0.666, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.666 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)  # neutral if undefined
    
    # Choppiness Index (CHOP) regime filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # avoid ranging markets
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators (KAMA, RSI, chop, volume are LTF, no HTF needed)
    # But we still align for consistency and proper min_periods handling
    kama_aligned = kama  # already LTF
    rsi_aligned = rsi
    chop_filter_aligned = chop_filter
    volume_confirm_aligned = volume_confirm
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.30   # Position size: 30% of capital
    
    # Warmup: need KAMA (1), RSI (14), chop (14), volume avg (20)
    start_idx = max(1, 14, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_filter_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        chop_ok = chop_filter_aligned[i]
        
        if position == 0:
            # Long: price > KAMA (uptrend) AND RSI < 30 (oversold bounce) AND volume + chop OK
            if close_val > kama_val and rsi_val < 30 and vol_conf and chop_ok:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short: price < KAMA (downtrend) AND RSI > 70 (overbought bounce) AND volume + chop OK
            elif close_val < kama_val and rsi_val > 70 and vol_conf and chop_ok:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: price < KAMA (trend change) OR RSI > 70 (overbought)
            if close_val < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price > KAMA (trend change) OR RSI < 30 (oversold)
            if close_val > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0