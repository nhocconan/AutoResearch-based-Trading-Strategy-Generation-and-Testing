#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_Trend
Hypothesis: On 4h timeframe, use daily KAMA to determine trend direction (bullish when price > KAMA, bearish when price < KAMA). Enter long on pullbacks when RSI(14) < 40 in bullish trend, enter short on bounces when RSI(14) > 60 in bearish trend. Use volume confirmation (>1.5x average) to filter false signals. Designed to work in both bull and bear markets by following the higher timeframe trend while entering on mean-reversion moves within that trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY INDICATORS: KAMA for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if len(close_1d) > 1 else 0
    # For simplicity, use a rolling calculation
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 10:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility if volatility > 0 else 0
        else:
            er[i] = 0
    
    sc = (er * 0.29 + 0.06) ** 2  # fast SC=2/(2+1), slow SC=2/(30+1)
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (13) + gain[i]) / 14 if i > 0 else gain[i]
                avg_loss[i] = (avg_loss[i-1] * (13) + loss[i]) / 14 if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend direction from daily KAMA
        bullish_trend = close[i] > kama_aligned[i]
        bearish_trend = close[i] < kama_aligned[i]
        
        # Entry conditions: pullback in trend with RSI extreme
        long_entry = bullish_trend and (rsi[i] < 40) and vol_confirm
        short_entry = bearish_trend and (rsi[i] > 60) and vol_confirm
        
        # Exit conditions: RSI returns to neutral zone
        exit_long = rsi[i] > 60
        exit_short = rsi[i] < 40
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals