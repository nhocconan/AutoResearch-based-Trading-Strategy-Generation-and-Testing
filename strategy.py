#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Camarilla_R3S3_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Camarilla levels (R3, S3) from previous day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's levels
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla calculations
        range_ = ph - pl
        camarilla_r3[i] = pc + range_ * 1.1 / 2
        camarilla_s3[i] = pc - range_ * 1.1 / 2
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_avg_today = vol_avg_20[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_current > 1.5 * vol_avg_today
        
        if position == 0:
            # Long: price breaks above R3 with volume and weekly uptrend
            if price > camarilla_r3[i] and vol_confirmed and price > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and weekly downtrend
            elif price < camarilla_s3[i] and vol_confirmed and price < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend changes
            if price < camarilla_s3[i] or price < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend changes
            if price > camarilla_r3[i] or price > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals