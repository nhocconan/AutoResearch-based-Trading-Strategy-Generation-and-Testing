#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume and ADX Trend Filter
Hypothesis: Donchian(20) breakouts capture momentum bursts. Volume and ADX ensure strong trends, reducing false signals. Designed for low trade frequency (~20-50/year) to minimize fee drag and improve generalization in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ADX and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate True Range for ATR
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ADX calculation
    plus_dm = np.where((high_daily - np.roll(high_daily, 1)) > (np.roll(low_daily, 1) - low_daily), 
                       np.maximum(high_daily - np.roll(high_daily, 1), 0), 0)
    minus_dm = np.where((np.roll(low_daily, 1) - low_daily) > (high_daily - np.roll(high_daily, 1)), 
                        np.maximum(np.roll(low_daily, 1) - low_daily, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_ma
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    tr_ma_aligned = align_htf_to_ltf(prices, df_daily, tr_ma)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(adx_aligned[i]) or np.isnan(tr_ma_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = tr_ma_aligned[i]
        adx_val = adx_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25
        trend_ok = adx_val > 25
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume and trend
            if price > upper and vol_ok and trend_ok:
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below lower Donchian with volume and trend
            elif price < lower and vol_ok and trend_ok:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or ATR-based stop
            if price < lower or (i > 0 and close[i-1] > lower and price < close[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or ATR-based stop
            if price > upper or (i > 0 and close[i-1] < upper and price > close[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Volume_ADXTrendFilter"
timeframe = "4h"
leverage = 1.0