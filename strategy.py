#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter Strategy
KAMA tracks trend direction, RSI provides entry timing, Chop filter avoids whipsaws in ranging markets.
Works in both bull and bear markets by adapting to trend strength via adaptive smoothing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    er_window = 10
    fast_ema = 2
    slow_ema = 30
    
    change = np.abs(np.diff(close, n=er_window))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 0 else np.array([])
    if len(volatility) == 0:
        er = np.zeros_like(change)
    else:
        # Pad volatility to match change length
        vol_padded = np.concatenate([np.full(er_window-1, np.nan), volatility])
        er = np.where(vol_padded[er_window-1:] != 0, change / vol_padded[er_window-1:], 0)
    
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    kama = np.full(n, np.nan)
    kama[er_window-1] = close[er_window-1]
    
    for i in range(er_window, n):
        if np.isnan(sc[i-er_window+1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i-er_window+1] * (close[i] - kama[i-1])
    
    # === RSI (Relative Strength Index) ===
    rsi_window = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[rsi_window] = np.nanmean(gain[:rsi_window])
    avg_loss[rsi_window] = np.nanmean(loss[:rsi_window])
    
    for i in range(rsi_window+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_window-1) + gain[i-1]) / rsi_window
        avg_loss[i] = (avg_loss[i-1] * (rsi_window-1) + loss[i-1]) / rsi_window
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Chopiness Index (Chop) ===
    chop_window = 14
    atr = np.full(n, np.nan)
    for i in range(chop_window, n):
        tr = np.max([
            high[i] - low[i],
            np.abs(high[i] - close[i-1]),
            np.abs(low[i] - close[i-1])
        ])
        if i == chop_window:
            atr[i] = np.nanmean([
                high[chop_window-1:chop_window] - low[chop_window-1:chop_window],
                np.abs(high[chop_window-1:chop_window] - close[chop_window-2:chop_window-1]),
                np.abs(low[chop_window-1:chop_window] - close[chop_window-2:chop_window-1])
            ])
        else:
            atr[i] = (atr[i-1] * (chop_window-1) + tr) / chop_window
    
    # True Range sum over window
    tr_sum = np.full(n, np.nan)
    for i in range(chop_window, n):
        tr_sum[i] = np.sum([
            high[i-chop_window+1:i+1] - low[i-chop_window+1:i+1],
            np.abs(high[i-chop_window+1:i+1] - close[i-chop_window:i]),
            np.abs(low[i-chop_window+1:i+1] - close[i-chop_window:i])
        ])
    
    # Max/min close over window
    max_close = np.full(n, np.nan)
    min_close = np.full(n, np.nan)
    for i in range(chop_window-1, n):
        max_close[i] = np.max(close[i-chop_window+1:i+1])
        min_close[i] = np.min(close[i-chop_window+1:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(chop_window-1, n):
        if max_close[i] - min_close[i] > 0 and not np.isnan(tr_sum[i]):
            chop[i] = 100 * np.log10(tr_sum[i] / (max_close[i] - min_close[i])) / np.log10(chop_window)
        else:
            chop[i] = 50  # neutral
    
    # === Weekly Trend Filter (Higher Timeframe) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly KAMA for trend
    wk_close = df_1w['close'].values
    wk_change = np.abs(np.diff(wk_close, n=10))
    wk_volatility = np.sum(np.abs(np.diff(wk_close)), axis=0) if len(wk_change) > 0 else np.array([])
    if len(wk_volatility) == 0:
        wk_er = np.zeros_like(wk_change)
    else:
        wk_vol_padded = np.concatenate([np.full(9, np.nan), wk_volatility])
        wk_er = np.where(wk_vol_padded[10:] != 0, wk_change / wk_vol_padded[10:], 0)
    
    wk_sc = (wk_er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    wk_kama = np.full(len(wk_close), np.nan)
    wk_kama[9] = wk_close[9]
    for i in range(10, len(wk_close)):
        if np.isnan(wk_sc[i-10]):
            wk_kama[i] = wk_kama[i-1]
        else:
            wk_kama[i] = wk_kama[i-1] + wk_sc[i-10] * (wk_close[i] - wk_kama[i-1])
    
    wk_kama_aligned = align_htf_to_ltf(prices, df_1w, wk_kama)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(50, chop_window, er_window, rsi_window)+1, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(wk_kama_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filters: price vs KAMA, weekly KAMA trend
        price_above_kama = close[i] > kama[i]
        wk_uptrend = wk_kama_aligned[i] > wk_kama_aligned[i-1] if i > 0 else True
        wk_downtrend = wk_kama_aligned[i] < wk_kama_aligned[i-1] if i > 0 else True
        
        # Chop filter: only trade when trending (Chop < 38.2) or extreme mean reversion (Chop > 61.8)
        trending_market = chop[i] < 38.2
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: trend weakness or reversal signal
            if not price_above_kama or not wk_uptrend or (rsi[i] > 70 and chop[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakness or reversal signal
            if price_above_kama or not wk_downtrend or (rsi[i] < 30 and chop[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long conditions: price above KAMA, weekly uptrend, RSI not overbought
            if price_above_kama and wk_uptrend and rsi[i] < 70:
                # Additional filters based on market regime
                if trending_market:
                    # In trend: enter on pullbacks (RSI < 40)
                    if rsi[i] < 40:
                        position = 1
                        signals[i] = 0.25
                elif ranging_market:
                    # In range: enter at extremes (RSI < 30 for long)
                    if rsi[i] < 30:
                        position = 1
                        signals[i] = 0.25
            
            # Short conditions: price below KAMA, weekly downtrend, RSI not oversold
            elif not price_above_kama and wk_downtrend and rsi[i] > 30:
                # Additional filters based on market regime
                if trending_market:
                    # In trend: enter on rallies (RSI > 60)
                    if rsi[i] > 60:
                        position = -1
                        signals[i] = -0.25
                elif ranging_market:
                    # In range: enter at extremes (RSI > 70 for short)
                    if rsi[i] > 70:
                        position = -1
                        signals[i] = -0.25
    
    return signals