#!/usr/bin/env python3
"""
1d_1W_KAMA_RSI_ChopFilter_V1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for overbought/oversold conditions and Choppiness Index to filter ranging markets.
Enter long when KAMA turns up (bullish) and RSI < 30 (oversold) in trending market (CHOP < 38.2).
Enter short when KAMA turns down (bearish) and RSI > 70 (overbought) in trending market.
Weekly trend filter: only take longs when price > weekly EMA20, shorts when price < weekly EMA20.
Position size: 0.25. Designed for low trade frequency (<15/year) to avoid fee drag.
Works in bull via KAMA trend + RSI pullbacks, in bear via short signals from overbought bounces.
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
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA) ===
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close_1d))
    change_sum = np.convolve(change, np.ones(9), mode='same')  # sum of 9 changes
    change_sum = np.concatenate([np.full(9, np.nan), change_sum[9:]])  # align to close index
    net_change = np.abs(np.concatenate([np.full(10, np.nan), np.diff(close_1d, n=10)]))
    er = np.where(change_sum > 0, net_change / change_sum, 0)
    
    # Smoothing constants: fast SC = 2/(2+1)=0.6667, slow SC = 2/(30+1)=0.0645
    sc = (er * 0.602 + 0.0645) ** 2  # smoothed ER scaled
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === RSI(14) ===
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='same')
    # Handle first values
    avg_gain[:13] = np.nan
    avg_loss[:13] = np.nan
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === CHOPPINESS INDEX (14) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    # Sum of TR over 14 periods
    tr_sum = np.convolve(tr, np.ones(14), mode='same')
    tr_sum[:13] = np.nan
    
    # Highest high and lowest low over 14 periods
    max_high = np.convolve(high_1d, np.ones(14), mode='same')
    max_high[:13] = np.nan
    min_low = np.convolve(low_1d, np.ones(14), mode='same')
    min_low[:13] = np.nan
    
    # Chop = 100 * log10(sumTR / (HH - LL)) / log10(14)
    hh_ll = max_high - min_low
    chop = np.where(hh_ll > 0, 100 * np.log10(tr_sum / hh_ll) / np.log10(14), 50)
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align daily indicators to 1d timeframe (no alignment needed for same TF)
    kama_aligned = kama
    rsi_aligned = rsi
    chop_aligned = chop
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trending market filter: Chop < 38.2 (trending), avoid ranging (Chop > 61.8)
        trending = chop_aligned[i] < 38.2
        
        # KAMA direction: turning point (slope change)
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        if trending:
            # Long: KAMA turning up + RSI oversold (<30) + price above weekly EMA20
            if kama_up and rsi_aligned[i] < 30 and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
            # Short: KAMA turning down + RSI overbought (>70) + price below weekly EMA20
            elif kama_down and rsi_aligned[i] > 70 and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # In ranging market, stay flat
            signals[i] = 0.0
    
    return signals

name = "1d_1W_KAMA_RSI_ChopFilter_V1"
timeframe = "1d"
leverage = 1.0