#!/usr/bin/env python3
# 1d_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend filter.
# In trending markets (price > KAMA), we look for RSI pullbacks to enter with momentum.
# In ranging markets (Choppiness Index > 61.8), we avoid trades to prevent whipsaws.
# This combines adaptive trend following with mean-reversion entries, working in both bull and bear markets
# by only taking trades aligned with the adaptive trend and avoiding choppy conditions.

name = "1d_KAMA_Trend_With_RSI_and_Chop_Filter"
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
    
    # Get weekly data for trend filter and chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    # KAMA: ER = |Change| / Sum|Δ|, SSC = [ER*(fastest- slowest) + slowest]^2
    # We'll use a simplified adaptive approach: fast EMA(2), slow EMA(30), weight based on volatility
    close_w = df_1w['close'].values
    # Calculate efficiency ratio
    change = np.abs(np.diff(close_w, prepend=close_w[0]))
    volatility = np.sum(np.abs(np.diff(close_w, prepend=close_w[0])), axis=0) if len(close_w) > 1 else 1
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    er = change / volatility
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fastest=2, slowest=30
    # Calculate KAMA
    kama = np.zeros_like(close_w)
    kama[0] = close_w[0]
    for i in range(1, len(close_w)):
        kama[i] = kama[i-1] + sc[i] * (close_w[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate weekly Choppiness Index for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    atr_w = np.zeros_like(close_w)
    tr_w = np.zeros_like(close_w)
    for i in range(len(close_w)):
        if i == 0:
            tr_w[i] = high[i] - low[i]
        else:
            tr_w[i] = max(high[i] - low[i], abs(high[i] - close_w[i-1]), abs(low[i] - close_w[i-1]))
        atr_w[i] = tr_w[i]  # Simple ATR for chop calculation (not smoothed)
    
    # Calculate chopping index over 14 periods
    chop = np.full_like(close_w, 50.0)  # Default to middle range
    for i in range(14, len(close_w)):
        atr_sum = np.sum(atr_w[i-13:i+1])
        hh = np.max(high[i-13:i+1])
        ll = np.min(low[i-13:i+1])
        if hh - ll > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50.0
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate daily RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (20), chop (14), RSI (14), volume MA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs weekly KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Regime filter: avoid choppy markets (CHOP > 61.8)
        ranging = chop_aligned[i] > 61.8
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + RSI pullback (30-50) + volume + not ranging
            if uptrend and 30 <= rsi[i] <= 50 and volume_confirm and not ranging:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + RSI bounce (50-70) + volume + not ranging
            elif downtrend and 50 <= rsi[i] <= 70 and volume_confirm and not ranging:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks, RSI overbought, or ranging market
            if not uptrend or rsi[i] > 70 or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks, RSI oversold, or ranging market
            if not downtrend or rsi[i] < 30 or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals