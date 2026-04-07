#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses Donchian(20) breakout on 12h timeframe for entry, 1d EMA(50) for trend filter,
# and volume spike (volume > 1.5x 20-period average) for confirmation.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at channel extremes.

name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume spike filter (volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper[i-1]  # break above previous upper band
        bearish_breakout = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Long: bullish breakout + uptrend + volume confirmation
        if bullish_breakout and uptrend and vol_confirm:
            signals[i] = 0.25
        # Short: bearish breakout + downtrend + volume confirmation
        elif bearish_breakout and downtrend and vol_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals