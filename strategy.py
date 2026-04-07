#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above Donchian(20) high and 1d EMA50 > EMA200
Short when price breaks below Donchian(20) low and 1d EMA50 < EMA200
Exit on opposite Donchian breakout or trend reversal
Designed to capture trends in both bull and bear markets with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
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
    
    # === Volume Filter (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < donch_low[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > donch_high[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish trend: EMA50 > EMA200
            if ema_50_aligned[i] > ema_200_aligned[i] and vol_ok:
                # Look for long entry on Donchian breakout
                if close[i] > donch_high[i]:
                    position = 1
                    signals[i] = 0.25
            # Bearish trend: EMA50 < EMA200
            elif ema_50_aligned[i] < ema_200_aligned[i] and vol_ok:
                # Look for short entry on Donchian breakout
                if close[i] < donch_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals