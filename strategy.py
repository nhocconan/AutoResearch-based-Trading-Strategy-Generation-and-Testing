#!/usr/bin/env python3
"""
4h_12h_RSI_Confluence_V1
Hypothesis: On 4h timeframe, buy when RSI(14) crosses above 30 with 12h trend filter (price > EMA50),
sell when RSI(14) crosses below 70 with 12h downtrend (price < EMA50). Exit on opposite RSI cross.
Uses volume confirmation to avoid false signals. Designed for low trade frequency (20-40/year)
by requiring RSI extremes + trend alignment + volume. Works in bull/bear via 12h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_RSI_Confluence_V1"
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
    
    # === 12h EMA50 FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h close
    ema_50 = np.zeros_like(close_12h)
    ema_50[:] = np.nan
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_12h)):
        if np.isnan(ema_50[i-1]) if i > 0 else True:
            ema_50[i] = close_12h[i]
        else:
            ema_50[i] = alpha * close_12h[i] + (1 - alpha) * ema_50[i-1]
    
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === RSI(14) ON 4h ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan
    
    for i in range(len(close)):
        if i < 14:
            continue
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME CONFIRMATION (20-period average) ===
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
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Entry conditions
        long_setup = (rsi[i] > 30) and (rsi[i-1] <= 30) and (close[i] > ema_50_aligned[i]) and vol_confirm
        short_setup = (rsi[i] < 70) and (rsi[i-1] >= 70) and (close[i] < ema_50_aligned[i]) and vol_confirm
        
        # Exit conditions: opposite RSI cross
        exit_long = (rsi[i] < 70) and (rsi[i-1] >= 70)
        exit_short = (rsi[i] > 30) and (rsi[i-1] <= 30)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
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