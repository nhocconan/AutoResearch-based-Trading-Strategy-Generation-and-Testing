#!/usr/bin/env python3
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
    
    # Load weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Load daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align weekly close for trend filter (no extra delay needed for EMA)
    ema_20_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        close_series = pd.Series(close_1w)
        ema_20_1w = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align pivot levels to daily timeframe
    pivot_d = align_htf_to_ltf(prices, df_1d, pivot)
    r1_d = align_htf_to_ltf(prices, df_1d, r1)
    s1_d = align_htf_to_ltf(prices, df_1d, s1)
    r2_d = align_htf_to_ltf(prices, df_1d, r2)
    s2_d = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 14-period daily ATR for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_d = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on daily)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_series = pd.Series(volume)
        vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_d[i]) or 
            np.isnan(r1_d[i]) or
            np.isnan(s1_d[i]) or
            np.isnan(r2_d[i]) or
            np.isnan(s2_d[i]) or
            np.isnan(atr_d[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.8% of price)
        if atr_d[i] < 0.008 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Weekly trend filter: only trade in direction of weekly EMA20
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R2 with volume confirmation AND weekly uptrend
            if (close[i] > r2_d[i] and volume_ratio > vol_threshold and weekly_uptrend):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S2 with volume confirmation AND weekly downtrend
            elif (close[i] < s2_d[i] and volume_ratio > vol_threshold and weekly_downtrend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S1
            if close[i] < s1_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R1
            if close[i] > r1_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Pivot_R2S2_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0