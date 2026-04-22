#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend direction + weekly RSI filter + volume confirmation.
# Uses KAMA's adaptive smoothing to capture trends while avoiding whipsaws in chop.
# Weekly RSI acts as a momentum filter to avoid counter-trend entries.
# Volume spike confirms institutional participation.
# Designed for low turnover (<20 trades/year) to minimize fee drag on 1d timeframe.
# Works in bull/bear by following adaptive trend with momentum filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for RSI filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate daily KAMA (adaptive trend)
    # Efficiency Ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # temporary, will compute properly below
    
    # Proper ER calculation
    price_change = np.abs(np.subtract(close[10:], close[:-10]))
    abs_diff = np.abs(np.diff(close))
    volatility_sum = np.convolve(abs_diff, np.ones(10), mode='valid')
    er = np.zeros_like(close)
    er[10:] = np.divide(price_change, volatility_sum, out=np.zeros_like(price_change), where=volatility_sum!=0)
    
    # Smoothing constants
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly RSI (14-period)
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
    
    # Align weekly RSI to daily (wait for weekly close)
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    
    # Volume average (20-day)
    vol_avg_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_avg_20[:10] = vol_avg_20[10]  # fill beginning
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # warmup for KAMA stability
        # Skip if weekly RSI not ready
        if np.isnan(rsi_weekly_aligned[i]):
            continue
        
        # Long conditions: price above KAMA (uptrend), RSI not overbought, volume confirmation
        if (close[i] > kama[i] and 
            rsi_weekly_aligned[i] < 70 and 
            volume[i] > 1.5 * vol_avg_20[i]):
            signals[i] = 0.25
        
        # Short conditions: price below KAMA (downtrend), RSI not oversold, volume confirmation
        elif (close[i] < kama[i] and 
              rsi_weekly_aligned[i] > 30 and 
              volume[i] > 1.5 * vol_avg_20[i]):
            signals[i] = -0.25
    
    return signals

name = "Daily_KAMA_WeeklyRSI_Volume_Filter"
timeframe = "1d"
leverage = 1.0