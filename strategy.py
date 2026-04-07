#!/usr/bin/env python3
"""
12h Donchian Breakout + 1d Trend Filter + Volume Spike + Chop Filter
Long when price breaks above Donchian upper band (20) and price > 1d EMA50; 
Short when price breaks below Donchian lower band (20) and price < 1d EMA50.
Exit when price crosses back through Donchian midline (20-period average).
Uses volume spike and chop filter (Choppiness Index > 61.8) for entry confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_chop_v1"
timeframe = "12h"
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
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    # Middle: average of upper and lower
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume Spike Detector (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # === Choppiness Index (14) - Range filter ===
    # CHOP > 61.8 = ranging market (good for mean reversion, but we avoid)
    # CHOP < 38.2 = trending market (good for breakouts)
    # We only take breakouts when CHOP < 61.8 (not extremely choppy)
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.maximum(np.abs(low - np.roll(close, 1)), 0)))
    # Handle first element
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Maximum range over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    max_range = max_high - min_low
    
    # Avoid division by zero
    chop = np.where(max_range > 0, 100 * np.log10(tr_sum / max_range) / np.log10(14), 50)
    chop_filter = chop < 61.8  # Avoid extremely choppy markets
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: break above upper band + price above 1d EMA50 + volume spike + not too choppy
            if (close[i] > donchian_high[i-1] and  # Break above previous bar's upper band
                close[i] > ema_50_aligned[i] and 
                vol_spike[i] and 
                chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: break below lower band + price below 1d EMA50 + volume spike + not too choppy
            elif (close[i] < donchian_low[i-1] and  # Break below previous bar's lower band
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i] and 
                  chop_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals