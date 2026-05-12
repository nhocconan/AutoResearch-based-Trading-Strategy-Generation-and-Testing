#!/usr/bin/env python3
name = "6h_WickReversal_With_TrendAndVolume"
timeframe = "6h"
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
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # === Weekly Bias Filter (1w) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Wick Reversal Detection ===
    body_size = np.abs(close - open_price) if 'open_price' in locals() else np.abs(close - np.roll(close, 1))
    if 'open_price' not in locals():
        open_price = prices['open'].values
    body_size = np.abs(close - open_price)
    upper_wick = high - np.maximum(close, open_price)
    lower_wick = np.minimum(close, open_price) - low
    total_range = high - low
    # Avoid division by zero
    total_range_safe = np.where(total_range == 0, 1e-10, total_range)
    lower_wick_ratio = lower_wick / total_range_safe
    upper_wick_ratio = upper_wick / total_range_safe
    
    # === Volume Spike Filter ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    # === ATR for Filtering (not used in signal size) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr[i]) or
            np.isnan(lower_wick_ratio[i]) or np.isnan(upper_wick_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish wick rejection at support + above 1d EMA20 + weekly bullish bias + volume spike
            if (lower_wick_ratio[i] > 0.6 and
                upper_wick_ratio[i] < 0.3 and
                close[i] > open_price[i] and  # bullish close
                close[i] > ema20_1d_aligned[i] and
                close[i] > ema50_1w_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish wick rejection at resistance + below 1d EMA20 + weekly bearish bias + volume spike
            elif (upper_wick_ratio[i] > 0.6 and
                  lower_wick_ratio[i] < 0.3 and
                  close[i] < open_price[i] and  # bearish close
                  close[i] < ema20_1d_aligned[i] and
                  close[i] < ema50_1w_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Wick rejection fails or weekly trend turns bearish
            if (upper_wick_ratio[i] > 0.5 and lower_wick_ratio[i] < 0.2) or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Wick rejection fails or weekly trend turns bullish
            if (lower_wick_ratio[i] > 0.5 and upper_wick_ratio[i] < 0.2) or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals