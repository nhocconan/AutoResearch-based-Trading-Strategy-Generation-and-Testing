#!/usr/bin/env python3
"""
1d_price_channel_1w_trend_volume_v1
Hypothesis: Price breaking above/below weekly Donchian channels (20-period) with volume confirmation
and weekly trend alignment captures institutional breakouts. Works in bull markets (breakouts continue)
and bear markets (breakdowns continue) by trading with the weekly trend. Uses 1d timeframe to balance
signal frequency and reduce fee drag. Targets 10-30 trades/year by requiring confluence of Donchian
break, volume spike, and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_price_channel_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly OHLC for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels from weekly high/low
    high_series = pd.Series(df_1w['high'].values)
    low_series = pd.Series(df_1w['low'].values)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (shift by 1 week for completed bars only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low OR trend turns down
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high OR trend turns up
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly Donchian high + volume + uptrend
            if (close[i] > donchian_high_aligned[i] and 
                vol_confirm and 
                close[i] > ema50_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly Donchian low + volume + downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_confirm and 
                  close[i] < ema50_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals