#!/usr/bin/env python3
# Hypothesis: 6h Donchian breakout with weekly RSI filter and volume confirmation
# Long when price breaks above 6h Donchian upper channel (20), weekly RSI > 50 (bullish bias), and volume spike
# Short when price breaks below 6h Donchian lower channel (20), weekly RSI < 50 (bearish bias), and volume spike
# Exit when price crosses the 6h Donchian midline or weekly RSI crosses 50 in opposite direction
# Weekly RSI acts as regime filter to avoid counter-trend trades in strong weekly trends
# Position size: 0.25 to limit drawdown. Target: 50-150 total trades over 4 years.

name = "6h_Donchian_WeeklyRSI_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Donchian high + weekly RSI > 50 + volume spike
            if (close[i] > donchian_high[i] and 
                rsi_1w_aligned[i] > 50 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian low + weekly RSI < 50 + volume spike
            elif (close[i] < donchian_low[i] and 
                  rsi_1w_aligned[i] < 50 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian mid OR weekly RSI < 50
            if (close[i] < donchian_mid[i]) or (rsi_1w_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian mid OR weekly RSI > 50
            if (close[i] > donchian_mid[i]) or (rsi_1w_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals