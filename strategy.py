#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_VolumeConfirm
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
filtered by RSI(14) for momentum strength, Choppiness Index to avoid ranging markets,
and volume confirmation to ensure institutional participation. Targets 10-15 trades/year
to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter (completed weekly close only)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on daily prices (ER=10, fast=2, slow=30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Choppiness Index (14)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / np.log10(max_high - min_low)) / np.log10(14)
    chop_values = chop
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly EMA50 for HTF trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA(10), RSI(14), CHOP(14), VOL(20)
    start_idx = max(10, 14, 14, 20, 50)  # +1 for weekly alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(chop_values[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI > 50 (momentum) + CHOP < 61.8 (trending) + volume spike + weekly uptrend
            long_setup = (close[i] > kama[i]) and \
                         (rsi_values[i] > 50) and \
                         (chop_values[i] < 61.8) and \
                         volume_spike[i] and \
                         (close[i] > ema_50_1w_aligned[i])
            # Short: price < KAMA (downtrend) + RSI < 50 (momentum) + CHOP < 61.8 (trending) + volume spike + weekly downtrend
            short_setup = (close[i] < kama[i]) and \
                          (rsi_values[i] < 50) and \
                          (chop_values[i] < 61.8) and \
                          volume_spike[i] and \
                          (close[i] < ema_50_1w_aligned[i])
            
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
            # Exit: price < KAMA OR RSI < 40 (weakening momentum) OR CHOP > 61.8 (ranging)
            if (close[i] < kama[i]) or \
               (rsi_values[i] < 40) or \
               (chop_values[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60 (weakening momentum) OR CHOP > 61.8 (ranging)
            if (close[i] > kama[i]) or \
               (rsi_values[i] > 60) or \
               (chop_values[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_VolumeConfirm"
timeframe = "1d"
leverage = 1.0