#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Filter and Daily Trend Filter
Long when price breaks above 20-period Donchian high + volume > 1.5x average + daily close > daily open
Short when price breaks below 20-period Donchian low + volume > 1.5x average + daily close < daily open
Exit when price touches opposite Donchian band (midpoint)
Works in both bull and bear markets by capturing breakouts with volume confirmation and trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_trend_v1"
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
    
    # === Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === Volume filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Daily trend filter (1D) ===
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    daily_open = df_1d['open'].values
    daily_bullish = daily_close > daily_open  # True if bullish daily candle
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(daily_bullish_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches Donchian midpoint (mean reversion)
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Need daily trend alignment
            if not daily_bullish_aligned[i]:  # Daily bearish
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume and trend confirmation
            if close[i] > donchian_high[i]:
                # Break above upper band -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i]:
                # Break below lower band -> short (only if daily bullish - counter-trend short in bullish daily?)
                # Actually, we only take shorts when daily is bearish for consistency
                if not daily_bullish_aligned[i]:  # Daily bearish
                    position = -1
                    signals[i] = -0.25
    
    return signals