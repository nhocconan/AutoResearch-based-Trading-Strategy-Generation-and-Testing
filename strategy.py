#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_RSI_Filter
# Hypothesis: Use 1d KAMA to determine trend direction on daily timeframe, filter with RSI for momentum strength.
# Trade weekly (1w) breakouts of KAMA-derived support/resistance levels with volume confirmation.
# Designed for low frequency (7-25 trades/year) to survive both bull and bear markets by following higher timeframe structure.

name = "1d_1w_KAMA_Trend_RSI_Filter"
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
    
    # === 1d KAMA for trend and levels ===
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # For efficiency ratio over 10 periods
    er = np.zeros_like(change)
    for i in range(len(change)):
        if i < 10:
            er[i] = 0
        else:
            num = np.abs(close[i] - close[i-10])
            den = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = num / den if den != 0 else 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1w KAMA for higher timeframe context ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    wk_close = df_1w['close'].values
    # Calculate weekly KAMA
    wk_change = np.abs(np.diff(wk_close, prepend=wk_close[0]))
    wk_volatility = np.sum(np.abs(np.diff(wk_close)), axis=0)
    wk_er = np.zeros_like(wk_change)
    for i in range(len(wk_change)):
        if i < 10:
            wk_er[i] = 0
        else:
            num = np.abs(wk_close[i] - wk_close[i-10])
            den = np.sum(np.abs(np.diff(wk_close[i-9:i+1])))
            wk_er[i] = num / den if den != 0 else 0
    
    wk_sc = (wk_er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    wk_kama = np.zeros_like(wk_close)
    wk_kama[0] = wk_close[0]
    for i in range(1, len(wk_close)):
        wk_kama[i] = wk_kama[i-1] + wk_sc[i] * (wk_close[i] - wk_kama[i-1])
    
    # Use previous week's KAMA for current week's support/resistance
    wk_kama_prev = np.roll(wk_kama, 1)
    wk_kama_prev[0] = np.nan
    
    # Weekly support/resistance bands around KAMA
    wk_atr = np.zeros_like(wk_close)
    for i in range(1, len(wk_close)):
        tr = max(
            wk_close[i] - wk_close[i-1],
            abs(wk_close[i] - wk_kama[i-1]),
            abs(wk_close[i-1] - wk_kama[i-1])
        )
        if i < 14:
            wk_atr[i] = np.mean(wk_atr[1:i]) if i > 1 else 0
        else:
            wk_atr[i] = (wk_atr[i-1] * 13 + tr) / 14
    
    wk_resistance = wk_kama_prev + wk_atr
    wk_support = wk_kama_prev - wk_atr
    
    # Align weekly levels to daily timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1w, wk_resistance)
    support_aligned = align_htf_to_ltf(prices, df_1w, wk_support)
    wk_kama_aligned = align_htf_to_ltf(prices, df_1w, wk_kama_prev)
    
    # Align daily KAMA
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    
    # === RSI filter (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (13) + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * (13) + loss[i]) / 14
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = rsi  # already on 1d timeframe
    
    # === Volume confirmation (20-day average) ===
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma_20[i] = np.mean(volume[max(0, i-19):i+1]) if i > 0 else volume[i]
        else:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(resistance_aligned[i]) or 
            np.isnan(support_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily KAMA
        trend_up = close[i] > kama_aligned[i]
        trend_down = close[i] < kama_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > resistance_aligned[i]
        breakout_down = close[i] < support_aligned[i]
        
        # RSI filter: not overbought/oversold
        rsi_ok = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: breakout above resistance, uptrend, RSI ok, volume confirmation
            if breakout_up and trend_up and rsi_ok and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below support, downtrend, RSI ok, volume confirmation
            elif breakout_down and trend_down and rsi_ok and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below support or trend reversal
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout above resistance or trend reversal
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals