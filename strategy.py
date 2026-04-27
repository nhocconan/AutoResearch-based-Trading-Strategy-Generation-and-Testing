#!/usr/bin/env python3
"""
6h_RSI_MultiTrend_Confluence_v1
Hypothesis: On 6b timeframe, combine RSI mean-reversion with multi-timeframe trend alignment (1d, 1w) to capture high-probability reversals in both bull and bear markets. Uses RSI(14) < 30 for long and > 70 for short, but only when aligned with higher timeframe trends. Volume confirmation filters low-conviction moves. Designed for low trade frequency (~20-50/year) to minimize fee drag.
"""

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
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA50 and EMA200 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1w data for longer-term trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all higher timeframe indicators
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need RSI (14), EMA50 (50), EMA200 (200), EMA50_1w (50), volume avg (20)
    start_idx = max(14, 50, 200, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema50_1d = ema50_1d_aligned[i]
        ema200_1d = ema200_1d_aligned[i]
        ema50_1w = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        # Trend alignment: bullish if price above both EMAs, bearish if below both
        bullish_alignment = close[i] > ema50_1d and close[i] > ema200_1d and close[i] > ema50_1w
        bearish_alignment = close[i] < ema50_1d and close[i] < ema200_1d and close[i] < ema50_1w
        
        if position == 0:
            # Long: RSI oversold in bullish alignment
            if rsi_val < 30 and bullish_alignment and vol_conf:
                signals[i] = size
                position = 1
            # Short: RSI overbought in bearish alignment
            elif rsi_val > 70 and bearish_alignment and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or trend breakdown
            if rsi_val > 70 or close[i] < ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or trend reversal
            if rsi_val < 30 or close[i] > ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_MultiTrend_Confluence_v1"
timeframe = "6h"
leverage = 1.0