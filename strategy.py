#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high + price above 1w EMA50 + volume > 1.5x average volume.
# Short when price breaks below Donchian(20) low + price below 1w EMA50 + volume > 1.5x average volume.
# Exit when price breaks back through Donchian levels (reverse breakout).
# Uses 1w EMA for trend filter to work in both bull and bear markets.
# Position size 0.25 to manage drawdown. Target 30-100 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    donch_upper = np.full(len(high_1d), np.nan)
    donch_lower = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_upper[i] = np.max(high_1d[i-20:i])
        donch_lower[i] = np.min(low_1d[i-20:i])
    
    # Get 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_upper[i-20]) or np.isnan(donch_lower[i-20]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_upper[i-20]
        lower = donch_lower[i-20]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Donchian Upper + above EMA50 + volume confirmation
            if (price > upper and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian Lower + below EMA50 + volume confirmation
            elif (price < lower and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian Lower or below EMA50
            if (price < lower or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian Upper or above EMA50
            if (price > upper or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_EMA_Volume"
timeframe = "1d"
leverage = 1.0