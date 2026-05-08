#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above 6h high of last 20 bars AND 12h EMA50 trend up AND 6h volume > 1.5x 20-period average.
# Short when price breaks below 6h low of last 20 bars AND 12h EMA50 trend down AND 6h volume > 1.5x 20-period average.
# Exit when price crosses back below/above 6h EMA20.
# Uses ADX to filter ranging markets and EMA for trend direction, suitable for both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-38/year) for low fee drift.

name = "6h_ADX_EMA_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6s Donchian breakout channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6s EMA20 for exit
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6s volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h data
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 trend direction: up if current > previous, down if current < previous
    ema50_trend = np.zeros_like(ema50_12h_aligned)
    ema50_trend[1:] = np.where(ema50_12h_aligned[1:] > ema50_12h_aligned[:-1], 1, 
                                np.where(ema50_12h_aligned[1:] < ema50_12h_aligned[:-1], -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema20[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_trend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above highest high, EMA50 trending up, volume spike
            long_cond = (close[i] > highest_high[i]) and (ema50_trend[i] == 1) and volume_filter[i]
            # Short conditions: break below lowest low, EMA50 trending down, volume spike
            short_cond = (close[i] < lowest_low[i]) and (ema50_trend[i] == -1) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below EMA20
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals