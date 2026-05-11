#!/usr/bin/env python3
# 4h_1D_RSI_Momentum_Squeeze
# Hypothesis: RSI momentum with Bollinger Band squeeze on 1d timeframe, traded on 4h.
# Long when: RSI(14) > 55 AND BB width (20,2) at 20-day low AND price > 4h EMA(50)
# Short when: RSI(14) < 45 AND BB width (20,2) at 20-day low AND price < 4h EMA(50)
# Exit when RSI crosses 50 or BB width expands above 40-day average.
# Works in bull by catching momentum in low volatility breakouts; works in bear by fading rallies in squeeze.
# Target: 25-40 trades/year (100-160 total) to avoid fee drag.

name = "4h_1D_RSI_Momentum_Squeeze"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for RSI and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d RSI(14) ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        if i < 14:
            avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else np.nan
            avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else np.nan
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # --- 1d Bollinger Bands (20,2) ---
    sma_20 = np.full_like(close_1d, np.nan)
    std_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i < 19:
            sma_20[i] = np.nan
            std_20[i] = np.nan
        else:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
            std_20[i] = np.std(close_1d[i-19:i+1])
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # BB width percentiles for squeeze detection
    bb_width_20d_low = np.full_like(bb_width, np.nan)
    bb_width_40d_avg = np.full_like(bb_width, np.nan)
    for i in range(len(bb_width)):
        if i < 19:
            bb_width_20d_low[i] = np.nan
            bb_width_40d_avg[i] = np.nan
        elif i < 39:
            bb_width_20d_low[i] = np.min(bb_width[20:i+1]) if i >= 20 else np.nan
            bb_width_40d_avg[i] = np.nan
        else:
            bb_width_20d_low[i] = np.min(bb_width[i-19:i+1])
            bb_width_40d_avg[i] = np.mean(bb_width[i-39:i+1])
    
    # --- 4h EMA(50) for trend filter ---
    ema_50 = np.full(n, np.nan)
    for i in range(n):
        if i < 49:
            ema_50[i] = np.nan
        elif i == 49:
            ema_50[i] = np.mean(close[0:50])
        else:
            ema_50[i] = (close[i] * 2 / (50 + 1)) + (ema_50[i-1] * 49 / (50 + 1))
    
    # Align 1d indicators to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bb_width_20d_low_aligned = align_htf_to_ltf(prices, df_1d, bb_width_20d_low)
    bb_width_40d_avg_aligned = align_htf_to_ltf(prices, df_1d, bb_width_40d_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1d RSI needs 14, BB 20, EMA50)
    start_idx = max(50, 40)  # EMA50 and BB40d
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(bb_width_20d_low_aligned[i]) or
            np.isnan(bb_width_40d_avg_aligned[i]) or
            np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width at 20-day low
        squeeze = bb_width_20d_low_aligned[i] <= bb_width_40d_avg_aligned[i] * 0.8
        
        if position == 0:
            if rsi_1d_aligned[i] > 55 and squeeze and close[i] > ema_50[i]:
                # Long: bullish momentum in low volatility
                signals[i] = 0.25
                position = 1
            elif rsi_1d_aligned[i] < 45 and squeeze and close[i] < ema_50[i]:
                # Short: bearish momentum in low volatility
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: RSI < 50 OR squeeze ends
                if rsi_1d_aligned[i] < 50 or bb_width_20d_low_aligned[i] > bb_width_40d_avg_aligned[i] * 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI > 50 OR squeeze ends
                if rsi_1d_aligned[i] > 50 or bb_width_20d_low_aligned[i] > bb_width_40d_avg_aligned[i] * 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals