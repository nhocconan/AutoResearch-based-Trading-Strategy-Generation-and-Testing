#!/usr/bin/env python3
"""
12h_RSI_25_75_Momentum_V1
Momentum strategy using RSI(14) on 12h timeframe with daily trend filter.
- Long: RSI crosses above 25 (from below) + daily EMA50 > EMA200
- Short: RSI crosses below 75 (from above) + daily EMA50 < EMA200
- Exit: Opposite RSI cross or trend reversal
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Works in bull markets (momentum continuation) and bear markets (mean reversion from extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA data to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate RSI(14) on 12h close prices
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # For periods where avg_loss is zero, RSI = 100
    rsi = np.where(avg_loss == 0, 100, rsi)
    # For periods where avg_gain is zero, RSI = 0
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # RSI conditions (using previous bar for crossover detection)
        rsi_prev = rsi[i-1]
        rsi_curr = rsi[i]
        rsi_cross_up_25 = (rsi_prev <= 25) and (rsi_curr > 25)
        rsi_cross_down_75 = (rsi_prev >= 75) and (rsi_curr < 75)
        
        if position == 0:
            # Long: uptrend + RSI crosses above 25 (from oversold)
            if uptrend and rsi_cross_up_25:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI crosses below 75 (from overbought)
            elif downtrend and rsi_cross_down_75:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or RSI crosses below 75 (overbought)
            if not uptrend or rsi_cross_down_75:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or RSI crosses above 25 (oversold)
            if not downtrend or rsi_cross_up_25:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_25_75_Momentum_V1"
timeframe = "12h"
leverage = 1.0