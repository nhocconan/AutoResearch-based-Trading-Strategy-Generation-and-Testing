#!/usr/bin/env python3
"""
4h Donchian breakout with volume confirmation and 1d trend filter
Long when price breaks above Donchian(20) high with volume surge in bullish regime
Short when price breaks below Donchian(20) low with volume surge in bearish regime
Exit when price crosses midline or Donchian opposite breakout
Designed to capture trends with controlled frequency
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # === Donchian Channels (20) ===
    # Highest high of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Midline
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === Volume Confirmation ===
    # Volume ratio: current volume / average volume of last 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(avg_volume > 0, volume / avg_volume, 0)
    
    # === 1d Trend Filter (EMA 50/200) ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime
        bullish_regime = ema_50_aligned[i] > ema_200_aligned[i]
        bearish_regime = ema_50_aligned[i] < ema_200_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: price crosses midline OR breakdown
            if close[i] <= donchian_mid[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price crosses midline OR breakout
            if close[i] >= donchian_mid[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume surge filter (at least 1.5x average volume)
            volume_surge = volume_ratio[i] > 1.5
            
            if bullish_regime and volume_surge:
                # Breakout above Donchian high
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
            elif bearish_regime and volume_surge:
                # Breakdown below Donchian low
                if close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals