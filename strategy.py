#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1w trend filter (EMA20) and 1d ATR-based breakout (ATR*2 from open)
# Trend filter avoids counter-trend trades; breakout captures momentum; volume filter reduces false signals
# Designed for low trade frequency (<30/year) to minimize fee drag, works in bull/bear via trend alignment

name = "12h_1w_trend_atr_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA20 on close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_20 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20[i] = (close_1w[i] * 2 + ema_20[i-1] * 18) / 20
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Daily ATR(14) for breakout threshold
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.zeros(len(df_1d))
    atr = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr[i] = high_1d[i] - low_1d[i]
        else:
            tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    atr[0] = tr[0]
    for i in range(1, len(df_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: 4-period average
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below open - ATR*2 OR trend turns bearish
            if (close[i] < (prices['open'].iloc[i] - 2 * atr_aligned[i]) or 
                close[i] < ema_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above open + ATR*2 OR trend turns bullish
            if (close[i] > (prices['open'].iloc[i] + 2 * atr_aligned[i]) or 
                close[i] > ema_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above open + ATR*2 with volume confirmation AND bullish trend
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > (prices['open'].iloc[i] + 2 * atr_aligned[i]) and 
                vol_ratio > 1.5 and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below open - ATR*2 with volume confirmation AND bearish trend
            elif (close[i] < (prices['open'].iloc[i] - 2 * atr_aligned[i]) and 
                  vol_ratio > 1.5 and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals