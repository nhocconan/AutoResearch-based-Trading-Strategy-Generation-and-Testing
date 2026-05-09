#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Uses 1h RSI for entry timing (overbought/oversold) in direction of 4h EMA trend.
# Long when RSI < 30 and price > 4h EMA200 with volume confirmation.
# Short when RSI > 70 and price < 4h EMA200 with volume confirmation.
# Designed to capture mean reversion in ranging markets while respecting higher timeframe trend.
# Session filter (08-20 UTC) reduces noise and improves win rate.
# Target: 15-37 trades/year per symbol to avoid fee drag.

name = "1h_RSI_4hEMA_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1h RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA200 trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 200)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price above 4h EMA200 + volume + session
            if (rsi[i] < 30 and close[i] > ema_4h_aligned[i] and 
                vol_confirm[i] and session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) + price below 4h EMA200 + volume + session
            elif (rsi[i] > 70 and close[i] < ema_4h_aligned[i] and 
                  vol_confirm[i] and session_filter[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or price below 4h EMA200
            if rsi[i] > 50 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or price above 4h EMA200
            if rsi[i] < 50 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals