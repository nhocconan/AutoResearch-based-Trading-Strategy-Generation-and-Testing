# 1h: 4h trend + 1d pivot breakout (long/short) with volume confirmation and session filter
# Uses 4h EMA for trend filter (avoid counter-trend trades) and 1d pivots for entry levels
# Target: 15-37 trades/year (60-150 total over 4 years) to avoid fee drag
# Works in bull/bear: trend filter + pivot breakouts capture momentum in trending markets, while volume confirmation filters false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4hEMA_1dPivot_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for pivot points (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align 1d pivot levels to 1h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_4h_aligned[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above EMA (uptrend) AND break above R1 with volume
            if price > ema_trend and price > r1 and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: price below EMA (downtrend) AND break below S1 with volume
            elif price < ema_trend and price < s1 and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price below EMA (trend change) OR below pivot (support break)
            if price < ema_trend or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price above EMA (trend change) OR above pivot (resistance break)
            if price > ema_trend or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals