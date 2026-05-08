#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 4-hour close above/below 200-period EMA with volume confirmation and weekly trend filter
# Long when price > EMA200, volume > 1.5x average volume, and weekly uptrend
# Short when price < EMA200, volume > 1.5x average volume, and weekly downtrend
# Uses EMA200 as strong trend filter, volume to confirm breakout strength, weekly trend for higher timeframe alignment
# Targets 20-50 trades per year to minimize fee drag while capturing strong trends

name = "4h_EMA200_Volume_WeeklyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # EMA200 on 4h close
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Average volume (50-period) for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        ema200_val = ema200[i]
        avg_vol = avg_volume[i]
        weekly_trend = ema34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        if position == 0:
            # Enter long: price > EMA200, volume confirmed, weekly uptrend
            if price > ema200_val and volume_confirmed and weekly_trend > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price < EMA200, volume confirmed, weekly downtrend
            elif price < ema200_val and volume_confirmed and weekly_trend < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < EMA200 or weekly trend down
            if price < ema200_val or weekly_trend < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > EMA200 or weekly trend up
            if price > ema200_val or weekly_trend > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals