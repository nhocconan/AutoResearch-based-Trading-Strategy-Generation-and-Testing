#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day RSI and weekly Bollinger Bands for mean reversion in extreme conditions.
# Long when 1d RSI < 30 and price touches weekly BB lower band with volume spike.
# Short when 1d RSI > 70 and price touches weekly BB upper band with volume spike.
# Exit when 1d RSI returns to neutral zone (40-60).
# Works in bull/bear by fading extremes only during high volatility periods.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1-day RSI(14)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # First average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    
    # Typical price for weekly
    tp_1w = (high_1w + low_1w + close_1w) / 3
    
    # Middle band (SMA)
    bb_middle_1w = np.full_like(tp_1w, np.nan)
    for i in range(bb_period - 1, len(tp_1w)):
        bb_middle_1w[i] = np.mean(tp_1w[i - bb_period + 1:i + 1])
    
    # Standard deviation
    bb_std_1w = np.full_like(tp_1w, np.nan)
    for i in range(bb_period - 1, len(tp_1w)):
        bb_std_1w[i] = np.std(tp_1w[i - bb_period + 1:i + 1])
    
    # Upper and lower bands
    bb_upper_1w = bb_middle_1w + (bb_std * bb_std_1w)
    bb_lower_1w = bb_middle_1w - (bb_std * bb_std_1w)
    
    # Volume moving average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bb_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_upper_1w)
    bb_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_lower_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI(14), BB(20), and volume MA20
    start_idx = max(rsi_period, bb_period, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(bb_upper_1w_aligned[i]) or 
            np.isnan(bb_lower_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg  # Require strong volume spike
        
        rsi = rsi_1d_aligned[i]
        bb_upper = bb_upper_1w_aligned[i]
        bb_lower = bb_lower_1w_aligned[i]
        
        if position == 0:
            # Long: RSI oversold + price at weekly BB lower + volume spike
            if (rsi < 30 and 
                price <= bb_lower * 1.001 and  # Allow tiny tolerance
                vol_filter):
                signals[i] = size
                position = 1
            # Short: RSI overbought + price at weekly BB upper + volume spike
            elif (rsi > 70 and 
                  price >= bb_upper * 0.999 and  # Allow tiny tolerance
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60)
            if 40 <= rsi <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60)
            if 40 <= rsi <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSIExtremes_WeeklyBB_Volume"
timeframe = "6h"
leverage = 1.0