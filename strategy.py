#!/usr/bin/env python3
"""
4h_price_channel_breakout_1d_trend_volume_v2
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with volume > 1.3x average during uptrend (price above 1d EMA50). Enter short when price breaks below Donchian(20) low with volume > 1.3x average during downtrend (price below 1d EMA50). Exit when price crosses the Donchian midline or trend changes. Uses 1d EMA50 trend filter and volume confirmation to avoid false breakouts. Target: 25-35 trades/year to minimize fee drag while capturing strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_breakout_1d_trend_volume_v2"
timeframe = "4h"
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
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.3x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.3)
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline or trend changes to downtrend
            if close[i] < donchian_mid[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline or trend changes to uptrend
            if close[i] > donchian_mid[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: break above Donchian high in uptrend (price above 1d EMA50)
                if (close[i] > high_roll[i] and 
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: break below Donchian low in downtrend (price below 1d EMA50)
                elif (close[i] < low_roll[i] and 
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals