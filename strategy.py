#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND 1w close > 1w open (bullish weekly candle) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND 1w close < 1w open (bearish weekly candle) AND volume > 1.5x average
# Exit when price touches opposite Donchian band or weekly trend reverses
# Targets 30-100 trades over 4 years by requiring multiple confluence factors

name = "1d_donchian_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (bullish/bearish candle)
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish weekly candle
    weekly_bearish = weekly_close < weekly_open  # True for bearish weekly candle
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or \
           np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or weekly_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with confluence: Donchian breakout + weekly trend + volume
            # Long: break above Donchian high + bullish weekly candle + volume confirmation
            if (close[i] > donchian_high[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + bearish weekly candle + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals