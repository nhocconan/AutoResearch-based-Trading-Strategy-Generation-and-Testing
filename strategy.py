#!/usr/bin/env python3
"""
4h_RSI_MeanReversion_Volume_Confirmation
Hypothesis: Uses RSI mean-reversion on 4h timeframe with volume spike confirmation and Bollinger Band squeeze filter. 
Enters long when RSI < 30 and price below lower Bollinger Band with volume > 1.5x average. 
Enters short when RSI > 70 and price above upper Bollinger Band with volume > 1.5x average. 
Uses 1d trend filter to align with higher timeframe direction. 
Designed for low trade frequency (15-30/year) with clear mean-reversion logic, works in ranging markets and avoids strong trends.
"""

name = "4h_RSI_MeanReversion_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-period, 2 std dev)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + (2 * std20)
    lower_bb = ma20 - (2 * std20)
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ma20[i]) or np.isnan(std20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_50_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: RSI oversold, price below lower BB, volume spike, daily trend up (for mean reversion in uptrend)
            if (rsi[i] < 30 and 
                close[i] < lower_bb[i] and 
                vol_ratio[i] > 1.5 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought, price above upper BB, volume spike, daily trend down (for mean reversion in downtrend)
            elif (rsi[i] > 70 and 
                  close[i] > upper_bb[i] and 
                  vol_ratio[i] > 1.5 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral or price reaches middle band
            if rsi[i] > 50 or close[i] > ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral or price reaches middle band
            if rsi[i] < 50 or close[i] < ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals