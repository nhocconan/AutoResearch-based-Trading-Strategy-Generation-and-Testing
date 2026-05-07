#!/usr/bin/env python3
# 4h_Volume_Weighted_RSI_Pullback_v2
# Hypothesis: Buy pullbacks in strong trends using RSI with volume-weighted smoothing.
# Uses volume-weighted RSI to filter weak moves, combined with daily trend filter (EMA50).
# Designed for 20-30 trades/year on 4h timeframe. Works in bull/bear via daily trend alignment.

name = "4h_Volume_Weighted_RSI_Pullback_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate volume-weighted RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted smoothing
    vol_weight = volume / (np.mean(volume) + 1e-8)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i] * vol_weight[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i] * vol_weight[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily trend filter (EMA50)
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Volume confirmation
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma20[:10] = np.nan
    vol_ma20[-10:] = np.nan
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: RSI pullback in uptrend with volume
            if (rsi[i] < 40 and 
                trend_up and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI bounce in downtrend with volume
            elif (rsi[i] > 60 and 
                  trend_down and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or trend reversal
            if rsi[i] > 70 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or trend reversal
            if rsi[i] < 30 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals