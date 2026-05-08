#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsVixFix_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Williams Vix Fix
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 22:
        return np.zeros(n)
    
    # Williams Vix Fix on weekly: measures fear/greed (low = fear, high = greed)
    highest_close = pd.Series(df_weekly['close']).rolling(window=22, min_periods=22).max().values
    wvf = ((highest_close - df_weekly['low'].values) / highest_close) * 100
    wvf_ma = pd.Series(wvf).rolling(window=10, min_periods=10).mean().values
    wvf_ma_aligned = align_htf_to_ltf(prices, df_weekly, wvf_ma)
    
    # Get daily trend for context
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend
    ema50_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume spike: current volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wvf_ma_aligned[i]) or np.isnan(ema50_daily_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wvf_val = wvf_ma_aligned[i]
        ema50_daily_val = ema50_daily_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: WVF low (fear) + price above daily EMA + volume spike (capitulation bounce)
            if (wvf_val < 30 and 
                close[i] > ema50_daily_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: WVF high (greed) + price below daily EMA + volume spike (exhaustion)
            elif (wvf_val > 80 and 
                  close[i] < ema50_daily_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: WVF rises above 50 (fear subsiding) OR price below daily EMA
            if (wvf_val > 50 or close[i] < ema50_daily_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: WVF falls below 50 (greed subsiding) OR price above daily EMA
            if (wvf_val < 50 or close[i] > ema50_daily_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals