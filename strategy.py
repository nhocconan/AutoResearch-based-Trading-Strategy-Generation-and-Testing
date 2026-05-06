#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using weekly RSI mean reversion with daily trend filter
# - Uses weekly RSI(14) for mean reversion signals (oversold/overbought)
# - Uses daily EMA(50) for trend direction filter
# - Uses 4h volume spike for entry confirmation
# - Enters long when weekly RSI < 30 and price > daily EMA50 with volume
# - Enters short when weekly RSI > 70 and price < daily EMA50 with volume
# - Exits when RSI returns to neutral zone (40-60)
# - Designed to work in both bull and bear markets by trading mean reversion within the trend
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_weeklyRSI_50EMA_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_weekly)
    avg_loss = np.zeros_like(close_weekly)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_weekly)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_weekly = 100 - (100 / (1 + rs))
    rsi_weekly = np.where(avg_loss == 0, 100, rsi_weekly)
    rsi_weekly = np.where(avg_gain == 0, 0, rsi_weekly)
    
    # Get daily data for EMA50
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50)
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly RSI and daily EMA50 to 4h timeframe
    rsi_weekly_4h = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    ema50_daily_4h = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume filter (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi_weekly_4h[i]) or np.isnan(ema50_daily_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly RSI oversold (<30) and price above daily EMA50 with volume
            if rsi_weekly_4h[i] < 30 and close[i] > ema50_daily_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI overbought (>70) and price below daily EMA50 with volume
            elif rsi_weekly_4h[i] > 70 and close[i] < ema50_daily_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>=40)
            if rsi_weekly_4h[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (<=60)
            if rsi_weekly_4h[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals