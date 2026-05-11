# 6h_1w_EquityCurveTrend_Filter
# Hypothesis: Use 1-week equity curve trend as a macro filter to determine long/short bias on 6-hour timeframe.
# Go long when 6h price > 6h SMA20 AND 1w equity curve trending up (price > 1w SMA50).
# Go short when 6h price < 6h SMA20 AND 1w equity curve trending down (price < 1w SMA50).
# Exit when price crosses 6h SMA20 OR 1w trend reverses.
# Equity curve trend acts as a regime filter: in long-term uptrend, buy dips; in downtrend, sell rallies.
# Works in bull by buying pullbacks in uptrend; works in bear by selling rallies in downtrend.
# Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.

name = "6h_1w_EquityCurveTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data for equity curve trend (using close as proxy)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 6h SMA20 ---
    sma_6h = np.full(n, np.nan)
    for i in range(20, n):
        sma_6h[i] = np.mean(close[i-20:i])
    
    # --- 1w SMA50 (equity curve trend proxy) ---
    close_1w = df_1w['close'].values
    sma_1w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-50:i])
    
    # Align 1w SMA to 6h
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(6h SMA20, 1w SMA50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(sma_6h[i]) or np.isnan(sma_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 6h price position relative to SMA20
        price_above_sma = close[i] > sma_6h[i]
        price_below_sma = close[i] < sma_6h[i]
        
        # Determine 1w trend (equity curve direction)
        trend_up = sma_1w_aligned[i] > 0  # Always true if not NaN, but we need actual trend
        # Actually check if current price is above/below 1w SMA50 for trend direction
        # We need the actual 1w close price aligned, not just SMA
        # Let's get 1w close aligned for direct comparison
    
    # Re-work: get 1w close aligned for trend determination
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        if np.isnan(sma_6h[i]) or np.isnan(close_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_above_sma = close[i] > sma_6h[i]
        price_below_sma = close[i] < sma_6h[i]
        
        # 1w trend: price above/below 1w SMA50
        # Need 1w SMA50 aligned
        if len(close_1w) >= 50:
            sma_1w_vals = np.full(len(close_1w), np.nan)
            for j in range(50, len(close_1w)):
                sma_1w_vals[j] = np.mean(close_1w[j-50:j])
            sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_vals)
        else:
            sma_1w_aligned = np.full(n, np.nan)
        
        if np.isnan(sma_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1w_aligned[i] > sma_1w_aligned[i]
        trend_down = close_1w_aligned[i] < sma_1w_aligned[i]
        
        if position == 0:
            if price_above_sma and trend_up:
                # Long: price above 6h SMA20 in 1w uptrend
                signals[i] = 0.25
                position = 1
            elif price_below_sma and trend_down:
                # Short: price below 6h SMA20 in 1w downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below 6h SMA20 OR 1w trend turns down
                if price_below_sma or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above 6h SMA20 OR 1w trend turns up
                if price_above_sma or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals