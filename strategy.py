#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R + 1d ADX regime
    # Strong trend (1d ADX > 25): trade pullbacks in trend direction using 6h Williams %R
    # Range (1d ADX < 20): fade extremes at Williams %R overbought/oversold
    # Uses discrete sizing 0.25 to minimize fee churn. Target: 12-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
        
        # Wilder's smoothing
        atr = np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)
        
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(n)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R(14)
    def williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(len(high)):
            if i < period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_6h = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d regime
        strong_trend = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if strong_trend:
            # Strong trend: trade pullbacks
            long_entry = wr_6h[i] < -80  # Oversold pullback in uptrend
            short_entry = wr_6h[i] > -20  # Overbought pullback in downtrend
        elif ranging:
            # Range: fade extremes
            long_entry = wr_6h[i] < -80  # Oversold bounce
            short_entry = wr_6h[i] > -20  # Overbought rejection
        
        # Exit logic: reverse signal
        long_exit = wr_6h[i] > -20  # Overbought
        short_exit = wr_6h[i] < -80  # Oversold
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_r_adx_regime_v1"
timeframe = "6h"
leverage = 1.0