#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1-week Donchian breakout with 1-day RSI momentum and volume confirmation.
# In bull markets: buy when price breaks above weekly Donchian high with RSI > 50 and volume > 1.5x average.
# In bear markets: sell when price breaks below weekly Donchian low with RSI < 50 and volume > 1.5x average.
# Uses weekly trend structure to avoid whipsaws, with daily momentum for entry timing.
# Volume filter ensures only high-conviction breaks. Target: 20-40 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend structure
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-period)
    donchian_len = 20
    if len(df_1w) < donchian_len:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels
    donchian_high = pd.Series(high_1w).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low_1w).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Align to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Load daily data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily RSI(14)
    rsi_len = 14
    if len(df_1d) < rsi_len:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_len, adjust=False, min_periods=rsi_len).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donchian_len*2, rsi_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high with bullish momentum
            if (close[i] > donchian_high_aligned[i] and 
                rsi_aligned[i] > 50 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short entry: price breaks below weekly Donchian low with bearish momentum
            elif (close[i] < donchian_low_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly Donchian midline or breaks below low
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if (close[i] < midline or 
                close[i] < donchian_low_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly Donchian midline or breaks above high
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if (close[i] > midline or 
                close[i] > donchian_high_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1w_Donchian_1d_RSI_Volume_v1"
timeframe = "4h"
leverage = 1.0