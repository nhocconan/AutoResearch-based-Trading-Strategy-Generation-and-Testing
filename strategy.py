#!/usr/bin/env python3
"""
12h_1w_KAMA_RSI_Trend_Momentum_v3
Hypothesis: Uses weekly KAMA direction with RSI momentum on 12h timeframe. Designed for low trade frequency (12-37/year) with trend-following in bull markets and mean-reversion in bear markets. Uses RSI extremes only when aligned with weekly KAMA trend to avoid whipsaws. Targets 20-40 total trades over 4 years per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_KAMA_RSI_Trend_Momentum_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on weekly data (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio and KAMA for weekly
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, k=1)), axis=0)  # 1-period volatility sum
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])  # Simplified
    
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    sc = (er * 0.28 + 0.06) ** 2  # 2 = fast SC, 30 = slow SC
    
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    if len(kama) > 0:
        kama[0] = close_1w[0]
        for i in range(1, len(kama)):
            if np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate RSI on 12h data (momentum)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder smoothing
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_lost, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(close[i]) or np.isnan(low[i]) or np.isnan(high[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend direction
        weekly_uptrend = close[i] > kama_aligned[i]
        weekly_downtrend = close[i] < kama_aligned[i]
        
        # RSI conditions: extreme values for mean reversion in ranging markets
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry conditions: trend following in strong trends, mean reversion in ranges
        # Only take counter-trend RSI signals when price is far from KAMA (strong trend)
        kama_distance = np.abs(close[i] - kama_aligned[i]) / kama_aligned[i]
        strong_trend = kama_distance > 0.05  # 5% deviation from KAMA
        
        long_entry = (weekly_uptrend and rsi[i] > 50 and rsi[i] < 60) or \
                     (not strong_trend and rsi_oversold)
        short_entry = (weekly_downtrend and rsi[i] < 50 and rsi[i] > 40) or \
                      (not strong_trend and rsi_overbought)
        
        # Exit conditions: opposite RSI extreme or trend change
        long_exit = rsi[i] > 70 or not weekly_uptrend
        short_exit = rsi[i] < 30 or not weekly_downtrend
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals