#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly EMA trend filter and volume confirmation
# Uses Donchian breakouts for trend following, weekly EMA for trend direction,
# and volume spike for confirmation. Designed for low trade frequency (target: 12-37/year)
# to minimize fee drag. Works in bull via breakouts and bear via mean reversion at extremes.

name = "12h_donchian20_weekly_ema_volume_v2"
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
    
    # Weekly data for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA(20)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily volume MA(20) for volume spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x daily MA
        volume_spike = volume[i] > 1.5 * volume_ma_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakdown_down = close[i] < lowest_low[i]
        
        # Long: bullish breakout with volume confirmation and above weekly EMA
        if breakout_up and volume_spike and above_ema:
            signals[i] = 0.25
        # Short: bearish breakdown with volume confirmation and below weekly EMA
        elif breakdown_down and volume_spike and below_ema:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals