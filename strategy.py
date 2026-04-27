#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-filtered 4h/1d regime-adaptive strategy.
# Uses 4h Donchian(20) + volume spike for trend direction, 1d RSI(14) for regime filter.
# Long when: price breaks above 4h Donchian upper band AND 4h volume > 1.5x 20-period avg AND 1d RSI(14) > 50.
# Short when: price breaks below 4h Donchian lower band AND 4h volume > 1.5x 20-period avg AND 1d RSI(14) < 50.
# Exit when: price crosses 4h Donchian midline OR RSI reverses (long: RSI<40, short: RSI>60).
# Session filter: 08-20 UTC only. Target 15-37 trades/year (60-150 total over 4 years).
# Uses Donchian breakouts for trend capture with volume confirmation and regime filter to avoid false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC only
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_len = 20
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(low_4h), np.nan)
    mid_4h = np.full(len(close_4h), np.nan)
    
    for i in range(donchian_len - 1, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i - donchian_len + 1:i + 1])
        lower_4h[i] = np.min(low_4h[i - donchian_len + 1:i + 1])
        mid_4h[i] = (upper_4h[i] + lower_4h[i]) / 2
    
    # Calculate 4h volume MA (20-period) for spike detection
    vol_ma_4h = np.full(len(volume_4h), np.nan)
    for i in range(19, len(volume_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i - 19:i + 1])
    
    # Align 4h indicators to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    mid_4h_aligned = align_htf_to_ltf(prices, df_4h, mid_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Get 1d data for RSI regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI (14-period)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    # Initial average
    if len(close_1d) >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(close_1d)):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.full(len(close_1d), np.nan)
    rsi_1d = np.full(len(close_1d), np.nan)
    for i in range(rsi_period - 1, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_1d[i] = 100.0 if avg_gain[i] > 0 else 0.0
    
    # Align 1d RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need Donchian(20), volume MA(20), RSI(14)
    start_idx = max(donchian_len - 1, 19, rsi_period - 1)
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(mid_4h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg_4h = vol_ma_4h_aligned[i]
        rsi_now = rsi_1d_aligned[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg_4h
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper band + volume spike + bullish regime (RSI>50)
            if (price > upper_4h_aligned[i] and vol_filter and rsi_now > 50):
                signals[i] = size
                position = 1
            # Short: price breaks below 4h Donchian lower band + volume spike + bearish regime (RSI<50)
            elif (price < lower_4h_aligned[i] and vol_filter and rsi_now < 50):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 4h Donchian midline OR RSI turns bearish (<40)
            if price < mid_4h_aligned[i] or rsi_now < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 4h Donchian midline OR RSI turns bullish (>60)
            if price > mid_4h_aligned[i] or rsi_now > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Session_Filtered_Donchian20_Volume_RSI14_Regime"
timeframe = "1h"
leverage = 1.0