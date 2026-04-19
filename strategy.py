#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above Donchian high(20) with price above weekly EMA20 and volume spike (>1.5x average).
# Short when price breaks below Donchian low(20) with price below weekly EMA20 and volume spike.
# Uses weekly EMA20 as trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
name = "12h_Donchian20_WeeklyEMA20_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA20 calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly close
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to 12h timeframe (wait for weekly close)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian channels (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high = high_roll[i]
        donch_low = low_roll[i]
        ema_trend = ema_20_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above weekly EMA20
            if price > donch_high and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below weekly EMA20
            elif price < donch_low and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below Donchian low or below weekly EMA20
            if price < donch_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above Donchian high or above weekly EMA20
            if price > donch_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals