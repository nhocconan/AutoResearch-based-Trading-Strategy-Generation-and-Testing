#!/usr/bin/env python3
# 1d_WeeklyCrossover_VolumeFilter
# Hypothesis: Weekly EMA crossover (13/34) with volume confirmation and 1d EMA200 trend filter
# Works in bull markets via bullish cross + uptrend, in bear markets via bearish cross + downtrend
# Volume filter reduces false signals, trend filter avoids counter-trend trades
# Target: 10-25 trades per year (~40-100 over 4 years) with position size 0.25

name = "1d_WeeklyCrossover_VolumeFilter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for EMA crossover
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Weekly EMA13 and EMA34 for crossover
    ema13_weekly = pd.Series(df_weekly['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34_weekly = pd.Series(df_weekly['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_weekly, ema13_weekly)
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Daily EMA200 for trend filter
    ema200_daily = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need 200 periods for EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(ema13_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(ema200_daily[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly EMA crossover signals
        bullish_cross = ema13_aligned[i] > ema34_aligned[i] and ema13_aligned[i-1] <= ema34_aligned[i-1]
        bearish_cross = ema13_aligned[i] < ema34_aligned[i] and ema13_aligned[i-1] >= ema34_aligned[i-1]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter from daily EMA200
        uptrend = close[i] > ema200_daily[i]
        downtrend = close[i] < ema200_daily[i]
        
        if position == 0:
            # Long: bullish cross + volume + uptrend
            if bullish_cross and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish cross + volume + downtrend
            elif bearish_cross and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish cross or trend reversal
            if bearish_cross or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish cross or trend reversal
            if bullish_cross or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals