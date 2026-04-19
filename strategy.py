#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# - 12h EMA(34) defines long-term trend direction (long when close > EMA34, short when close < EMA34)
# - 4h Donchian(20) breakout: long when close breaks above upper band, short when breaks below lower band
# - Volume filter: current 4h volume > 1.5x 20-period average for conviction
# - Exit when price crosses back through the Donchian midpoint (mean of upper/lower bands)
# - Position size: 0.25 (25%) to manage drawdown while capturing trends
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_Donchian20_12hTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 4h Donchian(20) channels
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 4h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x average
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: uptrend (close > 12h EMA34) + Donchian breakout + volume
            if close[i] > ema_34_12h_aligned[i] and close[i] > highest_high[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (close < 12h EMA34) + Donchian breakdown + volume
            elif close[i] < ema_34_12h_aligned[i] and close[i] < lowest_low[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals