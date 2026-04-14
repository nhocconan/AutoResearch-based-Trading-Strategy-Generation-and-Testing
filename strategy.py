#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Price Channel Breakout with Volume Confirmation and Daily Trend Filter
# Uses daily SMA(50) to determine long-term trend direction (bullish/bearish)
# Trades breakouts above/below 4h Donchian channels (20-period) in the direction of daily trend
# Volume > 1.3x average confirms breakout strength
# Exits when price returns to the midpoint of the Donchian channel (mean reversion within trend)
# Designed to work in both bull and bear markets by following the higher timeframe trend
# Target: 20-40 trades/year per symbol to minimize fee drag while maintaining edge

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily SMA(50) for trend direction
    sma_len = 50
    close_1d = df_1d['close'].values
    sma_1d = pd.Series(close_1d).rolling(window=sma_len, min_periods=sma_len).mean().values
    
    # Align daily SMA to 4h timeframe
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # 4h Donchian Channel (20-period)
    dc_len = 20
    upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().values
    lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().values
    
    # Midpoint for exit signal
    midpoint = (upper + lower) / 2
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma_1d_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(midpoint[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily SMA
        bullish_trend = close[i] > sma_1d_aligned[i]
        bearish_trend = close[i] < sma_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + bullish trend + volume
            if (close[i] > upper[i-1] and 
                bullish_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + bearish trend + volume
            elif (close[i] < lower[i-1] and 
                  bearish_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint (mean reversion)
            if close[i] < midpoint[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midpoint (mean reversion)
            if close[i] > midpoint[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_PriceChannel_Breakout_Volume_DailyTrend_v1"
timeframe = "4h"
leverage = 1.0