#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Filter_Chop_Regime
Hypothesis: KAMA (Kaufman Adaptive Moving Average) trend direction combined with RSI overbought/oversold levels and Choppiness Index regime filter.
In trending markets (CHOP < 38.2), follow KAMA direction with RSI confirmation.
In ranging markets (CHOP > 61.8), mean-revert at RSI extremes.
Designed for low trade frequency (7-25/year) on daily timeframe to minimize fee drag.
Works in both bull and bear markets by adapting to regime.
"""

name = "1d_KAMA_Direction_RSI_Filter_Chop_Regime"
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
    
    # === KAMA Calculation (10-period ER, 2/30 fast/slow SC) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper efficiency ratio calculation
    er = np.zeros(n)
    for i in range(n):
        if i == 0:
            er[i] = 0
        else:
            price_change = np.abs(close[i] - close[i-10] if i >= 10 else close[i] - close[0])
            volatility_sum = np.sum(np.abs(np.diff(close[max(0, i-9):i+1])))
            er[i] = price_change / (volatility_sum + 1e-10)
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI Calculation (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing
    for i in range(n):
        if i < 13:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 13:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14-period) ===
    atr = np.full(n, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    for i in range(n):
        if i < 13:
            atr[i] = np.nan
        elif i == 13:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.full(n, np.nan)
    for i in range(n):
        if i < 13:
            atr_sum[i] = np.nan
        elif i == 13:
            atr_sum[i] = np.sum(tr[0:14])
        else:
            atr_sum[i] = atr_sum[i-1] - tr[i-14] + tr[i]
    
    # True range max-min over 14 periods
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(n):
        if i < 13:
            max_high[i] = np.nan
            min_low[i] = np.nan
        elif i == 13:
            max_high[i] = np.max(high[0:14])
            min_low[i] = np.min(low[0:14])
        else:
            max_high[i] = max(max_high[i-1], high[i])
            min_low[i] = min(min_low[i-1], low[i])
    
    chop = np.full(n, np.nan)
    valid = (~np.isnan(atr_sum)) & (max_high - min_low > 0)
    chop[valid] = 100 * np.log10(atr_sum[valid] / (max_high[valid] - min_low[valid])) / np.log10(14)
    
    # === Weekly Trend Filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Ensure KAMA, RSI, CHOP, and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Determine regime
            is_trending = chop[i] < 38.2
            is_ranging = chop[i] > 61.8
            
            if is_trending:
                # Trending market: follow KAMA direction with RSI confirmation
                if close[i] > kama[i] and rsi[i] > 50 and rsi[i] < 70:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                elif close[i] < kama[i] and rsi[i] < 50 and rsi[i] > 30:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
            elif is_ranging:
                # Ranging market: mean reversion at RSI extremes
                if rsi[i] < 30 and close[i] > kama[i]:  # Oversold but price above KAMA (bullish bias)
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                elif rsi[i] > 70 and close[i] < kama[i]:  # Overbought but price below KAMA (bearish bias)
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit conditions
                is_trending = chop[i] < 38.2
                is_ranging = chop[i] > 61.8
                
                if is_trending:
                    # Exit trend follow: KAMA cross or RSI overbought
                    if close[i] < kama[i] or rsi[i] > 70:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.25
                else:
                    # Exit mean reversion: RSI returns to neutral or opposite extreme
                    if rsi[i] > 50 or rsi[i] < 30:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit conditions
                is_trending = chop[i] < 38.2
                is_ranging = chop[i] > 61.8
                
                if is_trending:
                    # Exit trend follow: KAMA cross or RSI oversold
                    if close[i] > kama[i] or rsi[i] < 30:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = -0.25
                else:
                    # Exit mean reversion: RSI returns to neutral or opposite extreme
                    if rsi[i] < 50 or rsi[i] > 70:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = -0.25
    
    return signals