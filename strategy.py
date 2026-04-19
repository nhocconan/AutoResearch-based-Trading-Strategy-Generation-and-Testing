#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# - 4h Donchian channels (20-period) for breakout signals
# - 12h EMA(34) for trend direction filter
# - 4h volume > 1.5x 20-period average for confirmation
# - Exit on opposite Donchian band touch or trend reversal
# - Position size: 0.25 to manage drawdown
# - Designed for fewer trades (target: 20-40/year) to minimize fee drag
# - Works in bull markets (breakouts) and bear markets (trend-following shorts)

name = "4h_Donchian20_12hTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 4h volume average (20-period)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average
        volume_filter = vol_ma_4h[i] > 0 and volume[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Look for long entry: price breaks above upper Donchian + uptrend + volume
            if (close[i] > highest_high[i-1] and  # Breakout above previous period's high
                close[i] > ema_34_12h_aligned[i] and  # Above 12h EMA (uptrend)
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below lower Donchian + downtrend + volume
            elif (close[i] < lowest_low[i-1] and  # Breakdown below previous period's low
                  close[i] < ema_34_12h_aligned[i] and  # Below 12h EMA (downtrend)
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown to lower Donchian or trend reversal
            if (close[i] < lowest_low[i] or  # Back below lower band
                close[i] < ema_34_12h_aligned[i]):  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on breakout above upper Donchian or trend reversal
            if (close[i] > highest_high[i] or  # Back above upper band
                close[i] > ema_34_12h_aligned[i]):  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals