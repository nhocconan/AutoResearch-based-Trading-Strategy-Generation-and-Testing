#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week Donchian channel breakout with volume confirmation and ATR filter
# Long when price breaks above weekly Donchian high with volume > 1.5x average and ATR > 0
# Short when price breaks below weekly Donchian low with volume > 1.5x average and ATR > 0
# Weekly Donchian provides strong structural levels, volume confirms breakout strength, ATR ensures volatility
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "4h_1wDonchian_Breakout_Volume_ATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-week Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Donchian high/low (20 periods)
    donch_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Volume confirmation: >1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma_30)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume confirmation
            if close[i] > donch_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donch_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals