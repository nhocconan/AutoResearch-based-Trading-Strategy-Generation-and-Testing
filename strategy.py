#!/usr/bin/env python3
name = "6h_ADX_Trend_With_Volume_Spike"
timeframe = "6h"
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
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def Wilder_smoothing(data, period):
            result = np.zeros_like(data)
            if len(data) < period:
                return result
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        plus_dm_smooth = Wilder_smoothing(plus_dm, period)
        minus_dm_smooth = Wilder_smoothing(minus_dm, period)
        tr_smooth = Wilder_smoothing(tr, period)
        
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        dx = np.zeros_like(close)
        dx = np.where((plus_di + minus_di) != 0, 
                     100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                     0)
        adx = Wilder_smoothing(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume filter: current volume > 2.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending) + volume spike + close > open (bullish bias)
            if (adx_12h_aligned[i] > 25 and 
                vol_filter[i] and 
                close[i] > prices['open'].iloc[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending) + volume spike + close < open (bearish bias)
            elif (adx_12h_aligned[i] > 25 and 
                  vol_filter[i] and 
                  close[i] < prices['open'].iloc[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX drops below 20 (trend weakening) or volume drops
            if adx_12h_aligned[i] < 20 or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX drops below 20 (trend weakening) or volume drops
            if adx_12h_aligned[i] < 20 or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals