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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR (14-period) for volatility filter
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily Donchian channels (20-period) from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donch_high_1d = np.full(len(df_1d), np.nan)
    donch_low_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            donch_high_1d[i] = np.max(high_1d[i-19:i+1])
            donch_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align weekly indicators to daily timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate volatility filter (weekly ATR > 1% of price)
    vol_filter_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if not np.isnan(atr_1d_aligned[i]) and close_1d[i] > 0:
            vol_filter_1d[i] = atr_1d_aligned[i] / close_1d[i] > 0.01
        else:
            vol_filter_1d[i] = False
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(donch_high_1d_aligned[i]) or
            np.isnan(donch_low_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1% of price)
        if vol_filter_1d[i] < 0.5:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA200
        uptrend = close_1d[i] > ema200_1d_aligned[i]
        downtrend = close_1d[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above daily Donchian high AND in uptrend
            if close_1d[i] > donch_high_1d_aligned[i] and uptrend:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low AND in downtrend
            elif close_1d[i] < donch_low_1d_aligned[i] and downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low OR trend changes
            if close_1d[i] < donch_low_1d_aligned[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high OR trend changes
            if close_1d[i] > donch_high_1d_aligned[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian20_EMA200_Trend_Filter_Vol"
timeframe = "1d"
leverage = 1.0