#!/usr/bin/env python3
"""
4h_RSI_40_60_MeanReversion_1dTrend_Volume
Hypothesis: Mean reversion on 4h using RSI 40-60 bounds (avoiding extremes) combined with 1d trend filter and volume confirmation. 
In bull markets (price > 1d EMA50): buy at RSI <= 40, sell at RSI >= 60.
In bear markets (price < 1d EMA50): sell at RSI >= 60, buy at RSI <= 40.
Volume surge (>1.5x 20-period average) confirms momentum for entry.
Target: 20-40 trades/year by using moderate RSI thresholds and requiring volume confirmation.
Works in both bull and bear by trading with the daily trend while using RSI for mean reversion entries.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align higher timeframe data to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for sufficient warmup (RSI + EMA)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price < EMA50 (bearish trend) AND RSI <= 40 (oversold) AND volume surge
        long_entry = (trend_down[i] and 
                     rsi_values[i] <= 40 and 
                     volume_surge[i])
        
        # Short: price > EMA50 (bullish trend) AND RSI >= 60 (overbought) AND volume surge
        short_entry = (trend_up[i] and 
                      rsi_values[i] >= 60 and 
                      volume_surge[i])
        
        # Exit when RSI returns to neutral zone (40-60) with volume surge
        long_exit = (rsi_values[i] >= 50 and volume_surge[i])  # Exit long when RSI >= 50
        short_exit = (rsi_values[i] <= 50 and volume_surge[i])  # Exit short when RSI <= 50
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_40_60_MeanReversion_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0