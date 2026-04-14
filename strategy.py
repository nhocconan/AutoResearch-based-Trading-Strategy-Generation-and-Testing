#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1-day Williams %R for overbought/oversold conditions
# with 4-hour RSI for momentum confirmation and volume spike for entry timing.
# Williams %R identifies extreme levels (>80 oversold, <20 overbought) on daily timeframe,
# RSI(14) > 50 confirms bullish momentum for longs, < 50 for shorts.
# Volume > 2x 20-period average ensures institutional participation.
# Designed to work in both bull and bear markets by fading extremes on higher timeframe.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 4h data ONCE for RSI and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate RSI on 4h data
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi_period = 14
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need volume MA and indicators
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND RSI > 50 (bullish momentum)
            if (williams_r_aligned[i] < -80 and 
                rsi_aligned[i] > 50 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) AND RSI < 50 (bearish momentum)
            elif (williams_r_aligned[i] > -20 and 
                  rsi_aligned[i] < 50 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or RSI < 40
            if (williams_r_aligned[i] > -50 or 
                rsi_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or RSI > 60
            if (williams_r_aligned[i] < -50 or 
                rsi_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dWilliamsR_4hRSI_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0