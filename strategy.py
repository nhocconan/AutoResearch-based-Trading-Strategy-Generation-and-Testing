#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d EMA200 trend filter
# Long when price breaks above 4h Donchian upper (20-period) with volume > 1.5x average and price above 1-day EMA200
# Short when price breaks below 4h Donchian lower (20-period) with volume > 1.5x average and price below 1-day EMA200
# Uses 4h for signal direction (structure), 1h only for entry timing precision
# Session filter (08-20 UTC) reduces noise trades
# Position size: 0.20 (20% of capital) to manage drawdown
# Target: 15-30 trades per year (60-120 over 4 years) to avoid fee drag

name = "1h_4hDonchian_20_1dEMA200_Volume_Breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channel (20-period) ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Donchian channels: 20-period high and low
    high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Upper and lower bands
    donchian_upper = high_20
    donchian_lower = low_20
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1-day EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: >1.5x 24-period average (6h equivalent)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian upper with volume and trend confirmation
            if close[i] > donchian_upper_aligned[i] and volume_filter[i] and close[i] > ema_200_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout: price breaks below Donchian lower with volume and trend confirmation
            elif close[i] < donchian_lower_aligned[i] and volume_filter[i] and close[i] < ema_200_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower (failed breakout) or trend turns bearish
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Donchian upper (failed breakdown) or trend turns bullish
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals