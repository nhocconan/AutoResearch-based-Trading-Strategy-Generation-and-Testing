#!/usr/bin/env python3
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
    
    # === 1w Donchian channel (20-period) for trend direction ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper/lower (20-period)
    upper_20 = np.full_like(high_1w, np.nan)
    lower_20 = np.full_like(low_1w, np.nan)
    for i in range(len(high_1w)):
        if i >= 19:
            upper_20[i] = np.max(high_1w[i-19:i+1])
            lower_20[i] = np.min(low_1w[i-19:i+1])
        elif i > 0:
            upper_20[i] = np.max(high_1w[0:i+1])
            lower_20[i] = np.min(low_1w[0:i+1])
        else:
            upper_20[i] = high_1w[0]
            lower_20[i] = low_1w[0]
    
    # Trend: price above upper = uptrend, below lower = downtrend
    trend_up = np.full_like(close_1w, False, dtype=bool)
    trend_down = np.full_like(close_1w, False, dtype=bool)
    close_1w = df_1w['close'].values
    for i in range(len(close_1w)):
        if not np.isnan(upper_20[i]) and not np.isnan(lower_20[i]):
            if close_1w[i] > upper_20[i]:
                trend_up[i] = True
            elif close_1w[i] < lower_20[i]:
                trend_down[i] = True
    
    # Align trend to 6h
    trend_up_6h = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    trend_down_6h = align_htf_to_ltf(prices, df_1w, trend_down.astype(float))
    
    # === 1d volume spike detection ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_1d[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_1d[0]
    
    # Volume spike: current volume > 2.0 x 20-period average
    volume_spike = volume_1d > (vol_ma_20 * 2.0)
    volume_spike_6h = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # === 6h RSI (14-period) for entry timing ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_up_6h[i]) or np.isnan(trend_down_6h[i]) or 
            np.isnan(volume_spike_6h[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: uptrend + volume spike + RSI < 40 (oversold)
            if (trend_up_6h[i] > 0.5 and 
                volume_spike_6h[i] > 0.5 and 
                rsi[i] < 40):
                signals[i] = 0.25
                position = 1
                continue
            # Short: downtrend + volume spike + RSI > 60 (overbought)
            elif (trend_down_6h[i] > 0.5 and 
                  volume_spike_6h[i] > 0.5 and 
                  rsi[i] > 60):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 60 (overbought) or trend changes
            if rsi[i] > 60 or trend_down_6h[i] > 0.5:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 40 (oversold) or trend changes
            if rsi[i] < 40 or trend_up_6h[i] > 0.5:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianTrend_VolumeSpike_RSI"
timeframe = "6h"
leverage = 1.0