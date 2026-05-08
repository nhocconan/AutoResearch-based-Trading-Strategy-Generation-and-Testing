#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (trend direction) with 1d volume spike and 1h RSI filter for entry timing.
# Uses 4h for trend direction (Donchian breakout), 1d for volume confirmation (2x 20-day EMA volume), and 1h RSI (14) to avoid overextended entries.
# Restricts to 08:00-20:00 UTC session to avoid low-liquidity periods. Position size fixed at 0.20 to manage risk.
# Designed for low trade frequency (~20-50/year) to minimize fee drag. Works in bull/bear by following 4h trend with strict 1h entry filters.

name = "1h_4hDonchian_1dVolume_1hRSI_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high_4h = rolling_max(high_4h, 20)
    donchian_low_4h = rolling_min(low_4h, 20)
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Get 1d data for volume spike (momentum confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Volume spike: 2x 20-day EMA
    vol_ema = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ema * 2.0)
    
    # Align 1d volume spike to 1h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Get 1h RSI for entry timing (avoid overextended entries)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        if len(gain) > period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi(close, 14)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_4h_aligned[i]) or 
            np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(rsi_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high + volume spike + RSI not overbought
            if close[i] > donchian_high_4h_aligned[i] and vol_spike_1d_aligned[i] and rsi_1h[i] < 70:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low + volume spike + RSI not oversold
            elif (i > 0 and close[i] < donchian_low_4h_aligned[i] and 
                  vol_spike_1d_aligned[i] and rsi_1h[i] > 30):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low or RSI overbought
            if close[i] < donchian_low_4h_aligned[i] or rsi_1h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high or RSI oversold
            if close[i] > donchian_high_4h_aligned[i] or rsi_1h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals