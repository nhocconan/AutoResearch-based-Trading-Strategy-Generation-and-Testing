#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# Volume spike (2.0x 20-period average) ensures strong participation
# ATR(14) trend filter: only long when price > SMA50 + 0.5*ATR, short when price < SMA50 - 0.5*ATR
# Discrete position sizing (0.25) minimizes fee drag while maintaining profitability
# Target: 100-180 total trades over 4 years = 25-45/year for 4h timeframe
# Works in both bull and bear markets by using volatility-adjusted trend filter

name = "4h_Donchian20_VolumeSpike_ATRTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR(14) for trend filter and volatility adjustment
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # SMA50 for trend direction
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Trend filter: price must be beyond 0.5*ATR from SMA50
    long_filter = close > (sma50 + 0.5 * atr)
    short_filter = close < (sma50 - 0.5 * atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(long_filter[i]) or 
            np.isnan(short_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian high + volume spike + bullish trend filter
            if close[i] > highest_high[i] and volume_spike[i] and long_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + volume spike + bearish trend filter
            elif close[i] < lowest_low[i] and volume_spike[i] and short_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Donchian low or bearish trend filter
            if close[i] < lowest_low[i] or not long_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Donchian high or bullish trend filter
            if close[i] > highest_high[i] or not short_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals