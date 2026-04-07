#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: Donchian channel breakout (20-period) on daily timeframe with volume confirmation and weekly EMA50 trend filter.
Enters long when price breaks above upper Donchian band with volume > 20-period average and price above weekly EMA50.
Enters short when price breaks below lower Donchian band with volume > 20-period average and price below weekly EMA50.
Designed for 10-25 trades/year to avoid fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower band
        
        # Weekly trend filter
        above_1w_ema50 = close[i] > ema50_1w_aligned[i]
        below_1w_ema50 = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or price below weekly EMA50
            if close[i] < donchian_lower[i] or below_1w_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or price above weekly EMA50
            if close[i] > donchian_upper[i] or above_1w_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above upper band with volume confirmation and above weekly EMA50
            if breakout_up and vol_confirmed and above_1w_ema50:
                position = 1
                signals[i] = 0.25
            # Short: breakout below lower band with volume confirmation and below weekly EMA50
            elif breakout_down and vol_confirmed and below_1w_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals