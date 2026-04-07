#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Volume Confirmation and 1w Trend Filter
Long when price breaks above Donchian(20) high + volume > 1.5x average + 1w close > SMA50
Short when price breaks below Donchian(20) low + volume > 1.5x average + 1w close < SMA50
Exit when price crosses opposite Donchian band or trend reverses
Designed to capture strong trends with volume confirmation in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_volume_1w_trend_v1"
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
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Average (20) ===
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # === 1w Trend Filter (SMA50) ===
    df_1w = get_htf_data(prices, '1w')
    sma_50 = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_avg[i]) or np.isnan(sma_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR trend reverses
            if close[i] < donch_low[i] or (sma_50_aligned[i] < close[i] and sma_50_aligned[i-1] >= close[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR trend reverses
            if close[i] > donch_high[i] or (sma_50_aligned[i] > close[i] and sma_50_aligned[i-1] <= close[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirmed = volume[i] > 1.5 * vol_avg[i]
            
            if vol_confirmed:
                # Long: break above Donchian high + 1w uptrend
                if close[i] > donch_high[i] and sma_50_aligned[i] > close[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: break below Donchian low + 1w downtrend
                elif close[i] < donch_low[i] and sma_50_aligned[i] < close[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals