# 1d_Camarilla_R1S1_Breakout_1wTrend_Volume
# Hypothesis: Trade breakouts of daily Camarilla R1/S1 levels only when aligned with weekly trend (EMA50) and confirmed by volume spike, on daily timeframe to capture major moves with low trade frequency (target 15-30 trades/year). Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaws in both bull and bear markets.

name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla pivot points (using prior day's OHLC)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_range = daily_high - daily_low
    r1 = daily_close + (camarilla_range * 1.1 / 12)
    s1 = daily_close - (camarilla_range * 1.1 / 12)
    
    # Align Camarilla levels to daily timeframe (with 1-day delay for completed bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=1)
    
    # Get daily volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
        if np.isnan(weekly_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = weekly_close_aligned[i] > ema_50_1w_aligned[i]
        trend_down = weekly_close_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with upward weekly trend and volume spike
            if (close[i] > r1_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with downward weekly trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily close or weekly trend turns down
            if close[i] < daily_close[i-1] or not trend_up:  # Use prior day's close for exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily close or weekly trend turns up
            if close[i] > daily_close[i-1] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals