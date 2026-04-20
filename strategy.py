#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Pivot Points (Daily)
    # Using standard formula: PP = (H + L + C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Weekly EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily pivot levels and weekly EMA to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume > 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else 0
        vol_filter = vol > vol_ma if vol_ma > 0 else False
        
        if position == 0:
            # Long: price above pivot, weekly uptrend, volume confirmation
            if price > pp_val and price < r1_val and ema_50_1w_val < price and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below pivot, weekly downtrend, volume confirmation
            elif price < pp_val and price > s1_val and ema_50_1w_val > price and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below pivot or weekly trend turns down
            if price < pp_val or ema_50_1w_val > price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above pivot or weekly trend turns up
            if price > pp_val or ema_50_1w_val < price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyPivot_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0