#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) determines trend direction on 1d,
# combined with RSI for overbought/oversold and Choppiness Index for regime filtering.
# Enters long when KAMA rising, RSI < 50, and CHOP > 61.8 (ranging market).
# Enters short when KAMA falling, RSI > 50, and CHOP > 61.8.
# Uses weekly trend filter to avoid counter-trend trades in strong trends.
# Designed for low-frequency, high-conviction trades in ranging markets.
# Target: 15-25 trades/year per symbol with disciplined risk management.

name = "1d_KAMA_RSI_ChopFilter"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 33) / 35
    
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on daily close
    def kama(close, er_period=10, fast_ema=2, slow_ema=30):
        n = len(close)
        kama_out = np.full(n, np.nan)
        if n < er_period:
            return kama_out
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.sum(np.abs(np.diff(close)), axis=0), '__len__') else np.sum(np.abs(np.diff(close)))
        # Manual calculation for efficiency ratio
        er = np.full(n, np.nan)
        for i in range(er_period, n):
            change_val = np.abs(close[i] - close[i-er_period])
            volatility_val = np.sum(np.abs(close[i-er_period+1:i+1] - close[i-er_period:i]))
            if volatility_val > 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = np.full(n, np.nan)
        for i in range(er_period, n):
            fast_sc = 2 / (fast_ema + 1)
            slow_sc = 2 / (slow_ema + 1)
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama_out[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            if not np.isnan(sc[i]):
                kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
            else:
                kama_out[i] = kama_out[i-1]
        return kama_out
    
    kama_vals = kama(close)
    
    # Calculate RSI (14)
    def rsi(close, period=14):
        n = len(close)
        rsi_out = np.full(n, np.nan)
        if n < period + 1:
            return rsi_out
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[period] = np.mean(gain[0:period])
        avg_loss[period] = np.mean(loss[0:period])
        
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.full(n, np.nan)
        for i in range(period, n):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
            else:
                rs[i] = 0
            if rs[i] != 0:
                rsi_out[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi_out[i] = 100
        return rsi_out
    
    rsi_vals = rsi(close)
    
    # Calculate Choppiness Index (14)
    def choppiness_index(high, low, close, period=14):
        n = len(close)
        chop = np.full(n, np.nan)
        if n < period:
            return chop
        
        atr = np.full(n, np.nan)
        for i in range(1, n):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if i == 1:
                atr[i] = tr
            else:
                atr[i] = (atr[i-1] * (period-1) + tr) / period
        
        for i in range(period, n):
            hh = np.max(high[i-period+1:i+1])
            ll = np.min(low[i-period+1:i+1])
            if hh - ll > 0:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / (hh - ll)) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop_vals = choppiness_index(high, low, close)
    
    # Align indicators (already daily, but ensure alignment for safety)
    kama_aligned = align_htf_to_ltf(prices, prices, kama_vals)  # identity alignment
    rsi_aligned = align_htf_to_ltf(prices, prices, rsi_vals)
    chop_aligned = align_htf_to_ltf(prices, prices, chop_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 14) + 1  # KAMA, RSI, CHOP periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Enter long: KAMA rising, RSI < 50, chop filter, and price above weekly EMA34
            if kama_aligned[i] > kama_aligned[i-1] and rsi_aligned[i] < 50 and chop_filter and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI > 50, chop filter, and price below weekly EMA34
            elif kama_aligned[i] < kama_aligned[i-1] and rsi_aligned[i] > 50 and chop_filter and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA falling OR RSI > 70 (overbought) OR chop < 38.8 (trending)
            if kama_aligned[i] < kama_aligned[i-1] or rsi_aligned[i] > 70 or chop_aligned[i] < 38.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA rising OR RSI < 30 (oversold) OR chop < 38.8 (trending)
            if kama_aligned[i] > kama_aligned[i-1] or rsi_aligned[i] < 30 or chop_aligned[i] < 38.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals