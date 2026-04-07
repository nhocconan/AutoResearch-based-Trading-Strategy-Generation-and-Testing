#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above 4h Donchian upper band and 1d EMA50 > EMA200 and volume > 20-period average
Short when price breaks below 4h Donchian lower band and 1d EMA50 < EMA200 and volume > 20-period average
Exit when price crosses the Donchian midpoint or trend reverses
Designed to capture breakouts in trending markets with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
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
    
    # === 4h Donchian Channel (20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === Volume Confirmation (20-period average) ===
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
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR trend reversal
            if close[i] < donch_mid[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR trend reversal
            if close[i] > donch_mid[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Bullish trend: EMA50 > EMA200
            if ema_50_aligned[i] > ema_200_aligned[i]:
                # Long breakout with volume confirmation
                if close[i] > donch_high[i] and volume[i] > vol_ma[i]:
                    position = 1
                    signals[i] = 0.30
            # Bearish trend: EMA50 < EMA200
            elif ema_50_aligned[i] < ema_200_aligned[i]:
                # Short breakdown with volume confirmation
                if close[i] < donch_low[i] and volume[i] > vol_ma[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals