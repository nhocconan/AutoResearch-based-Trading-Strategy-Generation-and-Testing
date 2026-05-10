#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Pullback
# Hypothesis: Use KAMA to determine trend direction on 1d timeframe, then look for RSI pullbacks on 1d for entry.
# In bull markets (KAMA rising): buy when RSI pulls back below 40.
# In bear markets (KAMA falling): sell when RSI rallies above 60.
# Uses weekly timeframe for regime filter: only trade when weekly KAMA aligns with daily trend.
# Target: 15-25 trades/year to stay under fee drag limits.

name = "1d_KAMA_Trend_RSI_Pullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily KAMA for trend
    def kama(close, length=10, fast=2, slow=30):
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if volatility[i-length+1:i+1].sum() != 0:
                er[i] = change[i-length+1:i+1].sum() / volatility[i-length+1:i+1].sum()
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        # Initialize KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # Daily RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Get daily data
    kama_daily = kama(close, 10, 2, 30)
    rsi_daily = rsi(close, 14)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    kama_weekly = kama(df_1w['close'].values, 10, 2, 30)
    kama_weekly_aligned = align_htf_to_ltf(prices, df_1w, kama_weekly)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama_daily[i]) or np.isnan(rsi_daily[i]) or 
            np.isnan(kama_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend from KAMA slope
        daily_trend_up = kama_daily[i] > kama_daily[i-1]
        daily_trend_down = kama_daily[i] < kama_daily[i-1]
        
        # Weekly trend alignment
        weekly_trend_up = kama_weekly_aligned[i] > kama_weekly_aligned[i-1] if i > 0 else False
        weekly_trend_down = kama_weekly_aligned[i] < kama_weekly_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: daily trend up, weekly trend aligned, RSI pullback
            if daily_trend_up and weekly_trend_up and rsi_daily[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily trend down, weekly trend aligned, RSI rally
            elif daily_trend_down and weekly_trend_down and rsi_daily[i] > 60 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: daily trend changes or RSI overbought
            if not daily_trend_up or rsi_daily[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: daily trend changes or RSI oversold
            if not daily_trend_down or rsi_daily[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals