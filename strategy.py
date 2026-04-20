# 6h_MeanReversion_RSI_BollingerBand_1DTrend
# Hypothesis: Mean-reversion on 6h timeframe using RSI + Bollinger Band reversals,
# filtered by daily trend to avoid counter-trend trades. Works in both bull and bear markets
# by only taking mean-reversion trades aligned with higher timeframe momentum.
# Target: 50-150 total trades over 4 years with position size 0.25.

name = "6h_MeanReversion_RSI_BollingerBand_1DTrend"
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
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema50_daily[i-1]
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Calculate Bollinger Bands (20, 2)
    sma20 = np.full(n, np.nan)
    std20 = np.full(n, np.nan)
    for i in range(20, n):
        sma20[i] = np.mean(close[i-20:i])
        std20[i] = np.std(close[i-20:i])
    
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(ema50_daily_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price touches lower band + daily uptrend
            if rsi[i] < 30 and close[i] <= lower_band[i] and close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price touches upper band + daily downtrend
            elif rsi[i] > 70 and close[i] >= upper_band[i] and close[i] < ema50_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) OR price reaches middle band
            if rsi[i] > 50 or close[i] >= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) OR price reaches middle band
            if rsi[i] < 50 or close[i] <= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals