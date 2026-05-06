#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d multi-timeframe confluence
# - Uses 4h Donchian breakout (20-period) for signal direction
# - Uses 1d EMA50 for trend filter (bullish above, bearish below)
# - Uses 1h volume expansion for entry timing
# - Enters only during 08-20 UTC session to avoid low-liquidity hours
# - Exits when price crosses 4h EMA21 (middle band)
# - Position size: 0.20
# - Target: 60-150 total trades over 4 years (15-37/year) with tight entry conditions

name = "1h_DonchianBreakout_4hEMA21_1dEMA50_Volume_Session"
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
    
    # Get 4h data for Donchian channels and EMA21
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_21_4h_1h = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (1h timeframe) - volume expansion
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_expansion = volume > (1.5 * vol_ma_10)  # Current volume > 1.5x 10-period average
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(ema_21_4h_1h[i]) or np.isnan(ema_50_1d_1h[i]) or 
            np.isnan(volume_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high + volume expansion + above 1d EMA50
            if (close[i] > donchian_high_1h[i] and 
                volume_expansion[i] and 
                close[i] > ema_50_1d_1h[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low + volume expansion + below 1d EMA50
            elif (close[i] < donchian_low_1h[i] and 
                  volume_expansion[i] and 
                  close[i] < ema_50_1d_1h[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA21
            if close[i] < ema_21_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA21
            if close[i] > ema_21_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals