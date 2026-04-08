#!/usr/bin/env python3
# 1d_weekly_macd_divergence_volume_v1
# Hypothesis: Trade weekly MACD divergences on daily timeframe with volume confirmation.
# In bull markets, buy bullish divergences; in bear markets, sell bearish divergences.
# Volume surge confirms divergence strength. Uses ATR-based stops to manage risk.
# Target: 20-50 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_macd_divergence_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly MACD (12,26,9)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate MACD components
    ema12 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align weekly MACD to daily
    macd_line_aligned = align_htf_to_ltf(prices, df_1w, macd_line)
    signal_line_aligned = align_htf_to_ltf(prices, df_1w, signal_line)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1w, macd_hist)
    
    # Daily ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: daily volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(macd_line_aligned[i]) or np.isnan(signal_line_aligned[i]) or 
            np.isnan(macd_hist_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Bearish divergence OR stoploss hit
            bearish_div = (high[i] > high[i-1] and macd_hist_aligned[i] < macd_hist_aligned[i-1])  # Higher price, lower MACD
            if bearish_div or close[i] < high[i-1] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bullish divergence OR stoploss hit
            bullish_div = (low[i] < low[i-1] and macd_hist_aligned[i] > macd_hist_aligned[i-1])  # Lower price, higher MACD
            if bullish_div or close[i] > low[i-1] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish divergence: lower price low, higher MACD low
            bullish_div = (low[i] < low[i-1] and macd_hist_aligned[i] > macd_hist_aligned[i-1])
            # Bearish divergence: higher price high, lower MACD high
            bearish_div = (high[i] > high[i-1] and macd_hist_aligned[i] < macd_hist_aligned[i-1])
            
            # Long entry: Bullish divergence with volume surge
            if bullish_div and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish divergence with volume surge
            elif bearish_div and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals