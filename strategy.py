#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
Long when price breaks above 20-period 4h Donchian upper channel with 4h volume > 1.5x 20-period average and 12h EMA34 trending up.
Short when price breaks below 20-period 4h Donchian lower channel with 4h volume > 1.5x 20-period average and 12h EMA34 trending down.
Exit when price returns to 20-period Donchian midpoint or reverses with volume confirmation.
Designed to capture breakouts with institutional volume participation in both bull and bear markets.
Volume confirmation reduces false breakouts, while 12h EMA34 filter ensures alignment with higher timeframe trend.
Target: 20-40 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 4h volume MA20 for confirmation
    volume_series = pd.Series(volume)
    vol_ma_20_4h = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian(20), volume MA20, and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20_4h[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper channel with volume confirmation and 12h uptrend
            if (close[i] > donchian_high[i] and 
                volume_confirmed and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower channel with volume confirmation and 12h downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_confirmed and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below Donchian midpoint OR breaks below lower channel with volume (reversal)
            if (close[i] <= donchian_mid[i] or 
                (close[i] < donchian_low[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above Donchian midpoint OR breaks above upper channel with volume (reversal)
            if (close[i] >= donchian_mid[i] or 
                (close[i] > donchian_high[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0