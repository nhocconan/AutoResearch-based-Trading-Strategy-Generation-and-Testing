#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_ChopFilter_VolumeConfirm
Hypothesis: On 12h timeframe, use KAMA for adaptive trend direction, RSI(14) for momentum filter,
Choppiness Index (CHOP) for regime detection, and volume confirmation to avoid false signals.
Long when KAMA upward, RSI > 50, CHOP < 61.8 (trending), and volume spike.
Short when KAMA downward, RSI < 50, CHOP < 61.8 (trending), and volume spike.
Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Targets 12-30 trades/year on 12h timeframe to minimize fee drag while capturing strong trends.
Works in both bull and bear markets by following the 1d EMA50 trend direction and using CHOP filter.
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
    
    # 1d EMA50 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of absolute changes
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Pad first element
    gain_padded = np.concatenate([[np.nan], gain])
    loss_padded = np.concatenate([[np.nan], loss])
    avg_gain = pd.Series(gain_padded).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss_padded).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) - 14 period
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 / (hh14 - ll14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((hh14 - ll14) != 0, chop, 50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align HTF indicators
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is 1d indicator
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)   # RSI is 1d indicator
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop) # CHOP is 1d indicator
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d indicators (EMA50, KAMA, RSI, CHOP) and volume MA
    start_idx = max(50, 14, 20) + 10  # EMA50 needs 50, KAMA/RSI/CHOP need 14, volume MA needs 20, plus KAMA ER lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA upward (price > KAMA), RSI > 50, CHOP < 61.8 (trending), volume spike, and 1d uptrend
            long_setup = (close[i] > kama_aligned[i]) and \
                         (rsi_aligned[i] > 50) and \
                         (chop_aligned[i] < 61.8) and \
                         volume_spike[i] and \
                         (close[i] > ema_50_1d_aligned[i])
            # Short: KAMA downward (price < KAMA), RSI < 50, CHOP < 61.8 (trending), volume spike, and 1d downtrend
            short_setup = (close[i] < kama_aligned[i]) and \
                          (rsi_aligned[i] < 50) and \
                          (chop_aligned[i] < 61.8) and \
                          volume_spike[i] and \
                          (close[i] < ema_50_1d_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: KAMA turns downward OR RSI < 40 OR CHOP > 61.8 (choppy) OR 1d trend turns down
            if (close[i] < kama_aligned[i]) or \
               (rsi_aligned[i] < 40) or \
               (chop_aligned[i] > 61.8) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA turns upward OR RSI > 60 OR CHOP > 61.8 (choppy) OR 1d trend turns up
            if (close[i] > kama_aligned[i]) or \
               (rsi_aligned[i] > 60) or \
               (chop_aligned[i] > 61.8) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_RSI_ChopFilter_VolumeConfirm"
timeframe = "12h"
leverage = 1.0