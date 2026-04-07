#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend and Volume Confirmation
Long when price breaks above Donchian(20) high with 12h EMA uptrend and volume spike
Short when price breaks below Donchian(20) low with 12h EMA downtrend and volume spike
Exit when price returns to Donchian midpoint or opposite breakout
Designed to capture breakouts in trending markets with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # === Donchian Channel (20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === 12h EMA Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False).mean().values
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Volume Spike Filter ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(volume_ma == 0, 1, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_20_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint or breaks below Donchian low
            if close[i] <= donch_mid[i] or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint or breaks above Donchian high
            if close[i] >= donch_mid[i] or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume spike confirmation
            vol_spike = volume_ratio[i] > 1.5
            
            # Uptrend: EMA20 > EMA50
            if ema_20_12h_aligned[i] > ema_50_12h_aligned[i]:
                # Look for long breakout
                if close[i] > donch_high[i] and vol_spike:
                    position = 1
                    signals[i] = 0.25
            # Downtrend: EMA20 < EMA50
            elif ema_20_12h_aligned[i] < ema_50_12h_aligned[i]:
                # Look for short breakout
                if close[i] < donch_low[i] and vol_spike:
                    position = -1
                    signals[i] = -0.25
    
    return signals