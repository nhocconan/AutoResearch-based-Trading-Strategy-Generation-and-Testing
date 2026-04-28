#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extreme readings with volume confirmation and 1d EMA50 trend filter.
# Enter long when Williams %R < -80 (oversold) with volume > 1.8x 24-bar average and price > 1d EMA50 (uptrend).
# Enter short when Williams %R > -20 (overbought) with volume > 1.8x 24-bar average and price < 1d EMA50 (downtrend).
# Exit on opposite Williams %R extreme or Donchian(10) break of the trend.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-150 trades over 4 years.
# Williams %R captures exhaustion points, volume confirms reversal pressure, 1d EMA50 filters counter-trend noise.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend) markets.

name = "6h_WilliamsR_1dEMA50_Volume_Extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA50 (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume confirmation: >1.8x 24-bar average volume (4d)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    # Calculate 6h Donchian(10) for exit
    def donchian_channels(high, low, length=10):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(length-1, len(high)):
            upper[i] = np.max(high[i-length+1:i+1])
            lower[i] = np.min(low[i-length+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions with volume confirmation and trend filter
        long_setup = williams_r_aligned[i] < -80 and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]
        short_setup = williams_r_aligned[i] > -20 and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions: opposite Williams %R extreme or Donchian break
        long_exit = williams_r_aligned[i] > -20 or close[i] < donchian_lower[i]
        short_exit = williams_r_aligned[i] < -80 or close[i] > donchian_upper[i]
        
        # Handle entries and exits
        if long_setup and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_setup and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals