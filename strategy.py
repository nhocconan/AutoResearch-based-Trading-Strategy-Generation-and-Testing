#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above Donchian(20) high, volume > 1.5x 20-period average, and price > 1d EMA50
# Short when price breaks below Donchian(20) low, volume > 1.5x 20-period average, and price < 1d EMA50
# Exit when price crosses back below Donchian(20) mean OR 1d EMA50 direction contradicts position
# Position size: 0.28 (28% of capital) to balance return and drawdown
# Designed to work in trending markets via EMA filter and reduce false breakouts with volume confirmation
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "4h_Donchian_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) for breakout signals
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Donchian(20) and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: break above Donchian high + volume spike + price above 1d EMA50
            if (close[i] > donchian_high[i] and 
                vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: break below Donchian low + volume spike + price below 1d EMA50
            elif (close[i] < donchian_low[i] and 
                  vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian mid OR price below 1d EMA50
            if (close[i] < donchian_mid[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price crosses above Donchian mid OR price above 1d EMA50
            if (close[i] > donchian_mid[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals