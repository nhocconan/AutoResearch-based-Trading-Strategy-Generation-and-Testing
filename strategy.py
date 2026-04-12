#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1w trend filter (EMA50) and volume confirmation
    # Uses Donchian channels from 12h for breakout levels: upper/lower 20-period
    # Trend filter: 1w EMA50 to ensure breakouts align with weekly direction
    # Volume confirmation: volume > 1.5 * 20-period average to avoid false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels on 12h data (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w trend
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Donchian breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Donchian high in bullish trend with volume
        if bullish_trend:
            long_entry = (close[i] > donchian_high[i]) and volume_spike[i]
        # Short breakout: price breaks below Donchian low in bearish trend with volume
        elif bearish_trend:
            short_entry = (close[i] < donchian_low[i]) and volume_spike[i]
        
        # Exit logic: opposite Donchian level or trend reversal
        long_exit = (bearish_trend and close[i] < donchian_low[i]) or \
                   (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > donchian_high[i]) or \
                    (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0