#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly RSI Reversion with 1d Trend Filter
# Go long when weekly RSI(14) crosses below 30 (oversold) and price > 1d EMA200 (bullish trend)
# Go short when weekly RSI(14) crosses above 70 (overbought) and price < 1d EMA200 (bearish trend)
# Exit when RSI returns to 50 (mean reversion complete)
# Weekly RSI avoids noise; 1d EMA200 ensures trend alignment to avoid counter-trend trades
# Target: 12-37 trades/year by requiring extreme weekly RSI + trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for RSI calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 14:
        return np.zeros(n)
    
    # Load daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_w = df_w['close'].values
    delta = np.diff(close_w, prepend=close_w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_w)
    avg_loss = np.zeros_like(close_w)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_w = 100 - (100 / (1 + rs))
    rsi_w[0:13] = np.nan  # Not enough data
    
    # Align weekly RSI to 6h
    rsi_w_aligned = align_htf_to_ltf(prices, df_w, rsi_w)
    
    # Calculate daily EMA200 for trend filter
    close_d = df_1d['close'].values
    ema200_d = pd.Series(close_d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to 6h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if np.isnan(rsi_w_aligned[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current values
        rsi = rsi_w_aligned[i]
        price = prices['close'].iloc[i]
        ema200 = ema200_1d_aligned[i]
        
        if position == 0:
            # Enter long: weekly RSI < 30 (oversold) and price above EMA200 (bullish trend)
            if rsi < 30 and price > ema200:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly RSI > 70 (overbought) and price below EMA200 (bearish trend)
            elif rsi > 70 and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI returns to 50 (mean reversion complete)
            if (position == 1 and rsi >= 50) or (position == -1 and rsi <= 50):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyRSI_Reversion_Trend"
timeframe = "6h"
leverage = 1.0