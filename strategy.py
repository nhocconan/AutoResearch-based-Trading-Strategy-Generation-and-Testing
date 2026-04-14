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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR (14-period)
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
    
    # Calculate weekly volume moving average (20-period)
    vol_ma_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        vol_ma_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(df_1w)):
            vol_ma_1w[i] = (vol_ma_1w[i-1] * 19 + volume_1w[i]) / 20
    
    # Calculate weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full(len(df_1w), np.nan)
    avg_loss = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(df_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if not np.isnan(avg_loss[i]) and avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_1w[i] = 100 - (100 / (1 + rs))
        elif not np.isnan(avg_gain[i]) and avg_loss[i] == 0:
            rsi_1w[i] = 100.0
    
    # Align indicators to daily timeframe
    atr_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    vol_ma_1d = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    rsi_1d = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d[i]) or
            np.isnan(vol_ma_1d[i]) or
            np.isnan(rsi_1d[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1.5% of price)
        if atr_1d[i] / close[i] < 0.015:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 20-period MA)
        if volume[i] < vol_ma_1d[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high AND weekly RSI > 50
            if close[i] > donch_high[i] and rsi_1d[i] > 50:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low AND weekly RSI < 50
            elif close[i] < donch_low[i] and rsi_1d[i] < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low OR weekly RSI < 40
            if close[i] < donch_low[i] or rsi_1d[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high OR weekly RSI > 60
            if close[i] > donch_high[i] or rsi_1d[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_RSI_Volume_Filter"
timeframe = "1d"
leverage = 1.0