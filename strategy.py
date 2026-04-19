#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when: price breaks above upper Donchian(20) + weekly EMA200 uptrend + volume > 1.5x 20-day average
# Short when: price breaks below lower Donchian(20) + weekly EMA200 downtrend + volume > 1.5x 20-day average
# Exit when price returns to the 20-day SMA or reverses to opposite Donchian band.
# Designed for low-frequency, high-conviction trades (~10-20/year) to minimize fee drag.
# Weekly EMA200 filter ensures alignment with long-term trend, avoiding counter-trend trades in choppy markets.
name = "1d_Donchian20_WeeklyEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily 20-period SMA for exit
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for weekly EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(sma_20[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_donchian = high_roll[i]
        lower_donchian = low_roll[i]
        sma = sma_20[i]
        ema_200 = ema_200_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper Donchian with volume confirmation and weekly uptrend
            if price > upper_donchian and vol > 1.5 * vol_ma and price > ema_200:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower Donchian with volume confirmation and weekly downtrend
            elif price < lower_donchian and vol > 1.5 * vol_ma and price < ema_200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to 20-day SMA or breaks below lower Donchian (reversal)
            if price <= sma or price < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 20-day SMA or breaks above upper Donchian (reversal)
            if price >= sma or price > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals