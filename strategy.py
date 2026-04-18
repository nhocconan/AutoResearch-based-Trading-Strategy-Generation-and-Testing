#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard)
    pivot_1w = np.full_like(close_1w, np.nan)
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    r2_1w = np.full_like(close_1w, np.nan)
    s2_1w = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i > 0 and not (np.isnan(high_1w[i-1]) or np.isnan(low_1w[i-1]) or np.isnan(close_1w[i-1])):
            pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
            r1_1w[i] = 2 * pivot_1w[i] - low_1w[i-1]
            s1_1w[i] = 2 * pivot_1w[i] - high_1w[i-1]
            r2_1w[i] = pivot_1w[i] + (high_1w[i-1] - low_1w[i-1])
            s2_1w[i] = pivot_1w[i] - (high_1w[i-1] - low_1w[i-1])
    
    # Calculate 20-period EMA on 1w for trend filter
    if len(close_1w) >= 20:
        ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_20_1w = np.full_like(close_1w, np.nan)
    
    # Calculate 14-period ATR on 1w for volatility filter
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.full_like(high, np.nan)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    vol_period = 20
    
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-vol_period:i])
    
    # Align all data to 6h timeframe (primary)
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    ema_20_1w_6h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    atr_1w_6h = align_htf_to_ltf(prices, df_1w, atr_1w)
    vol_ma_1d_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 20, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_6h[i]) or np.isnan(r1_1w_6h[i]) or np.isnan(s1_1w_6h[i]) or 
            np.isnan(r2_1w_6h[i]) or np.isnan(s2_1w_6h[i]) or np.isnan(ema_20_1w_6h[i]) or 
            np.isnan(atr_1w_6h[i]) or np.isnan(vol_ma_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (1d)
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_6h[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_20_1w_6h[i]
        downtrend = close[i] < ema_20_1w_6h[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1w_6h[i] > 0.005 * close[i]  # ATR > 0.5% of price
        
        if position == 0:
            # Long: price breaks above weekly R2 with uptrend and volume
            if close[i] > r2_1w_6h[i] and uptrend and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S2 with downtrend and volume
            elif close[i] < s2_1w_6h[i] and downtrend and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly S1 OR trend reverses
            if close[i] < s1_1w_6h[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly R1 OR trend reverses
            if close[i] > r1_1w_6h[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_EMA20_Volume"
timeframe = "6h"
leverage = 1.0