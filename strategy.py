#!/usr/bin/env python3
"""
6h Donchian Breakout with 12h Trend Filter and Volume Confirmation
Long when price breaks above Donchian(20) high and 12h EMA50 > EMA200 with volume > 1.5x average
Short when price breaks below Donchian(20) low and 12h EMA50 < EMA200 with volume > 1.5x average
Exit on opposite Donchian break or trend reversal
Designed to capture strong trends with volume confirmation in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_12h_trend_volume_v1"
timeframe = "6h"
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h EMA Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # === Volume Average (20-period) ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or \
           np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < lowest_low[i] or ema_50_12h_aligned[i] < ema_200_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > highest_high[i] or ema_50_12h_aligned[i] > ema_200_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Bullish trend: EMA50 > EMA200
            if ema_50_12h_aligned[i] > ema_200_12h_aligned[i] and vol_confirm:
                # Long: price breaks above Donchian high
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
            # Bearish trend: EMA50 < EMA200
            elif ema_50_12h_aligned[i] < ema_200_12h_aligned[i] and vol_confirm:
                # Short: price breaks below Donchian low
                if close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals