#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly RSI extremes with daily trend filter and volume confirmation
# - Uses weekly RSI(14) for extreme overbought/oversold conditions (<30 or >70)
# - Uses daily EMA(50) to filter trend direction (long when price > EMA, short when price < EMA)
# - Uses 12h volume spike for entry confirmation
# - Designed to capture mean reversions in strong trends with institutional participation
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_weeklyRSI_50EMA_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_weekly = df_weekly['close'].values
    delta = np.diff(close_weekly, prepend=close_weekly[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_weekly)
    avg_loss = np.zeros_like(close_weekly)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_weekly)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_weekly = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 12h timeframe
    rsi_weekly_12h = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    
    # Get daily data for EMA
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50)
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).values
    
    # Align daily EMA to 12h timeframe
    ema_50_12h = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume filter (12h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi_weekly_12h[i]) or np.isnan(ema_50_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly RSI oversold (<30) + price above daily EMA50 + volume spike
            if rsi_weekly_12h[i] < 30 and close[i] > ema_50_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI overbought (>70) + price below daily EMA50 + volume spike
            elif rsi_weekly_12h[i] > 70 and close[i] < ema_50_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly RSI returns to neutral (50) or price crosses below EMA
            if rsi_weekly_12h[i] >= 50 or close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly RSI returns to neutral (50) or price crosses above EMA
            if rsi_weekly_12h[i] <= 50 or close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals