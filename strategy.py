# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR-based volatility filter and volume confirmation
# This strategy trades breakouts of the 20-period Donchian channel when volatility
# is elevated (ATR ratio > 1.5) and volume is above average. It uses the 1d EMA(34)
# as a trend filter to avoid counter-trend trades. Designed to work in both bull
# and bear markets by following the higher timeframe trend.
# Uses discrete position sizing (0.25) to limit turnover and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels, EMA trend, and ATR (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    # Upper = max(high_1d, lookback=20)
    # Lower = min(low_1d, lookback=20)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian upper + ATR ratio > 1.5 + volume spike + above EMA
            if (close[i] > donchian_upper_aligned[i] and 
                atr_ratio_aligned[i] > 1.5 and 
                volume[i] > 1.5 * vol_avg_20[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower + ATR ratio > 1.5 + volume spike + below EMA
            elif (close[i] < donchian_lower_aligned[i] and 
                  atr_ratio_aligned[i] > 1.5 and 
                  volume[i] > 1.5 * vol_avg_20[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian midpoint or ATR ratio drops below 1.0
            midpoint = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2
            if position == 1:
                # Exit long: Price below midpoint or low volatility
                if close[i] < midpoint or atr_ratio_aligned[i] < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price above midpoint or low volatility
                if close[i] > midpoint or atr_ratio_aligned[i] < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_ATR_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0