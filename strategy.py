#!/usr/bin/env python3
"""
1h_4d_Trend_Momentum_v1
Hypothesis: Use 4h trend (price above/below 4h EMA50) and 1h momentum (RSI(14) > 60 for long, < 40 for short) with volume confirmation (volume > 1.5x 20-period average). Enter on momentum pullbacks in trending direction. Exit on opposite momentum extreme. Designed for 15-30 trades/year on 1h timeframe by requiring trend alignment and momentum extremes. Works in bull via long bias in uptrend, works in bear via short capability in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Trend_Momentum_v1"
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
    
    # === 4H EMA(50) FOR TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    if len(close_4h) >= 50:
        ema_50_4h = np.zeros_like(close_4h)
        ema_50_4h[0] = close_4h[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_4h)):
            ema_50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_50_4h[i-1]
    else:
        ema_50_4h = np.full_like(close_4h, np.nan)
    
    # Align 4h EMA to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1H RSI(14) FOR MOMENTUM ===
    if len(close) >= 14:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.full_like(close, 50.0)
    
    # === VOLUME AVERAGE (20-period) ===
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
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Momentum conditions
        rsi_overbought = rsi[i] > 60
        rsi_oversold = rsi[i] < 40
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions: momentum pullback in trend direction
        long_setup = price_above_ema and rsi_oversold and vol_confirm
        short_setup = price_below_ema and rsi_overbought and vol_confirm
        
        # Exit conditions: opposite momentum extreme
        exit_long = rsi[i] > 70  # overbought exit for long
        exit_short = rsi[i] < 30  # oversold exit for short
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals