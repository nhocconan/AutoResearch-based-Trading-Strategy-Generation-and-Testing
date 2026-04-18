#!/usr/bin/env python3
"""
12h Donchian Channel Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above 12h Donchian upper channel (20-period) with 1d EMA50 uptrend and volume spike.
Short when price breaks below 12h Donchian lower channel with 1d EMA50 downtrend and volume spike.
Uses 1d EMA50 for trend filter to avoid counter-trend trades, improving performance in both bull and bear markets.
Designed for low trade frequency (target: 12-37 trades/year) with strong trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channels (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, donchian_period)  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        price_above_upper = price > upper_channel[i]
        price_below_lower = price < lower_channel[i]
        
        # Determine 1d trend: above EMA50 = uptrend, below = downtrend
        uptrend = ema_50_1d_aligned[i] > 0 and price > ema_50_1d_aligned[i]
        downtrend = ema_50_1d_aligned[i] > 0 and price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike and 1d uptrend
            if price_above_upper and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume spike and 1d downtrend
            elif price_below_lower and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: maintain until reversal signal
            signals[i] = 0.25
            # Exit: price breaks below lower channel (trend reversal)
            if price_below_lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: maintain until reversal signal
            signals[i] = -0.25
            # Exit: price breaks above upper channel (trend reversal)
            if price_above_upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0