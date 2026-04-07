#!/usr/bin/env python3
"""
12h_price_channel_breakout_volume_v1
Hypothesis: On 12h timeframe, enter long when price breaks above Donchian(20) high with volume > 1.5x average, enter short when price breaks below Donchian(20) low with volume > 1.5x average. Exit when price crosses the Donchian midline or reverses with strong volume. Use 1d trend filter (price above/below EMA50) to avoid counter-trend trades. Designed for 12-37 trades/year to minimize fee drift while capturing breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_price_channel_breakout_volume_v1"
timeframe = "12h"
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
    
    # Donchian Channel (20-period)
    if len(high) < 20:
        return np.zeros(n)
    
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_mid = (dc_high + dc_low) / 2.0
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(dc_mid[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline OR strong bearish reversal
            if close[i] < dc_mid[i] or (close[i] < close[i-1] and volume[i] > vol_ma[i] * 2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline OR strong bullish reversal
            if close[i] > dc_mid[i] or (close[i] > close[i-1] and volume[i] > vol_ma[i] * 2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian high with 1d uptrend
                if close[i] > dc_high[i] and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with 1d downtrend
                elif close[i] < dc_low[i] and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals