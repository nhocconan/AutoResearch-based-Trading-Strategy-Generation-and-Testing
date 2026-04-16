#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Donchian breakout + 1d volume filter + RSI filter
# Long when price breaks above 4h Donchian upper channel (20) AND 1d volume > 1.5x average AND RSI < 60
# Short when price breaks below 4h Donchian lower channel (20) AND 1d volume > 1.5x average AND RSI > 40
# Uses 4h for signal direction (Donchian breakout), 1h only for entry timing
# Volume filter ensures conviction, RSI filter avoids overextended moves
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag
# Session filter: 08-20 UTC to reduce noise

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === 1d volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values  # 30 days average
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === RSI(14) on 1h ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        rsi_val = rsi[i]
        
        # Volume confirmation: current volume > 1.5x 1d average
        vol_confirm = volume[i] > vol_ma * 1.5
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices.iloc[i]['open_time']).hour
        in_session = 8 <= hour <= 20
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price touches middle of Donchian channel or RSI > 70
            mid = (upper + lower) / 2
            if price <= mid or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit when price touches middle of Donchian channel or RSI < 30
            mid = (upper + lower) / 2
            if price >= mid or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and in_session:
            # Long when: price breaks above Donchian upper AND volume confirmation AND RSI < 60
            if price > upper and vol_confirm and rsi_val < 60:
                signals[i] = 0.20
                position = 1
                continue
            # Short when: price breaks below Donchian lower AND volume confirmation AND RSI > 40
            elif price < lower and vol_confirm and rsi_val > 40:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_1dVolume1.5x_RSIFilter"
timeframe = "1h"
leverage = 1.0