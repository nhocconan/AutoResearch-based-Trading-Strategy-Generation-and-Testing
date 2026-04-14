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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA 50 for trend
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    upper_20 = np.full(len(df_1d), np.nan)
    lower_20 = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            upper_20[i] = np.max(high_1d[i-19:i+1])
            lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian channels to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Daily ATR for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]) if i-1 >= 0 else high_1d[i] - low_1d[i],
            abs(low_1d[i] - close_1d[i-1]) if i-1 >= 0 else high_1d[i] - low_1d[i]
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on daily)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_aligned[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Trend filter: only trade in direction of weekly EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume confirmation and uptrend
            if (close[i] > upper_20_aligned[i] and 
                volume_ratio > vol_threshold and 
                trend_up):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian with volume confirmation and downtrend
            elif (close[i] < lower_20_aligned[i] and 
                  volume_ratio > vol_threshold and 
                  trend_down):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below lower Donchian
            if close[i] < lower_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above upper Donchian
            if close[i] > upper_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_Volume_Trend"
timeframe = "1d"
leverage = 1.0