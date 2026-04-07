#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend and Volume Confirmation
Long when price breaks above Donchian(20) high in bullish regime (1d EMA50 > EMA200) with volume > 1.5x avg
Short when price breaks below Donchian(20) low in bearish regime (1d EMA50 < EMA200) with volume > 1.5x avg
Exit when price crosses opposite Donchian band or regime changes
Designed to capture strong trends with filtered breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # === Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ratio[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR regime turns bearish
            if close[i] < donch_low[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR regime turns bullish
            if close[i] > donch_high[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Bullish regime: EMA50 > EMA200
            if ema_50_aligned[i] > ema_200_aligned[i]:
                # Look for long breakout with volume confirmation
                if close[i] > donch_high[i] and vol_ratio[i] > 1.5:
                    position = 1
                    signals[i] = 0.30
            # Bearish regime: EMA50 < EMA200
            elif ema_50_aligned[i] < ema_200_aligned[i]:
                # Look for short breakdown with volume confirmation
                if close[i] < donch_low[i] and vol_ratio[i] > 1.5:
                    position = -1
                    signals[i] = -0.30
    
    return signals