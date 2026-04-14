#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data for additional context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr_1w = np.zeros(len(df_1w))
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr_1w[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    atr_6h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly moving average (10-period) for trend
    ma_10_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 10:
        for i in range(9, len(df_1w)):
            ma_10_1w[i] = np.mean(close_1w[i-9:i+1])
    
    ma_10_6h = align_htf_to_ltf(prices, df_1w, ma_10_1w)
    
    # Calculate weekly RSI (14-period) for overbought/oversold
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(df_1w), np.nan)
    avg_loss = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(df_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_6h = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily pivot points (standard)
    # Using previous day's data for today's pivot
    pivot_1d = np.full(len(df_1d), np.nan)
    r1_1d = np.full(len(df_1d), np.nan)
    s1_1d = np.full(len(df_1d), np.nan)
    r2_1d = np.full(len(df_1d), np.nan)
    s2_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1 = 2 * pp - low_1d[i-1]
        s1 = 2 * pp - high_1d[i-1]
        r2 = pp + (high_1d[i-1] - low_1d[i-1])
        s2 = pp - (high_1d[i-1] - low_1d[i-1])
        pivot_1d[i] = pp
        r1_1d[i] = r1
        s1_1d[i] = s1
        r2_1d[i] = r2
        s2_1d[i] = s2
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate 6-hour Donchian channels (20-period) for entry timing
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate volume moving average (20-period) for confirmation
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_6h[i]) or
            np.isnan(ma_10_6h[i]) or
            np.isnan(rsi_6h[i]) or
            np.isnan(pivot_6h[i]) or
            np.isnan(r1_6h[i]) or
            np.isnan(s1_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.4% of price)
        if atr_6h[i] < 0.004 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 1.8
        
        # Trend filter: price above/below weekly MA
        above_weekly_ma = close[i] > ma_10_6h[i]
        below_weekly_ma = close[i] < ma_10_6h[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_not_extreme = (rsi_6h[i] > 20) and (rsi_6h[i] < 80)
        
        if position == 0:
            # Long: Price breaks above 6h Donchian high with volume confirmation,
            # above weekly MA, and not overbought
            if (close[i] > donch_high[i] and 
                volume_ratio > vol_threshold and 
                above_weekly_ma and 
                rsi_not_extreme):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 6h Donchian low with volume confirmation,
            # below weekly MA, and not oversold
            elif (close[i] < donch_low[i] and 
                  volume_ratio > vol_threshold and 
                  below_weekly_ma and 
                  rsi_not_extreme):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR below weekly MA
            if close[i] < donch_low[i] or close[i] < ma_10_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR above weekly MA
            if close[i] > donch_high[i] or close[i] > ma_10_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_MA_RSI_Volume"
timeframe = "6h"
leverage = 1.0