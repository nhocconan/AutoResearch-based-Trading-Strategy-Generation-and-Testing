#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Donchian channel breakout (20-period) + volume confirmation + 12h EMA34 trend filter.
Long when price breaks above 12h Donchian upper band with volume > 1.5x 20-period average and price > 12h EMA34.
Short when price breaks below 12h Donchian lower band with volume > 1.5x 20-period average and price < 12h EMA34.
Exit when price returns to the opposite Donchian band or reverses against trend.
Designed to capture strong momentum moves with institutional structure from 12h timeframe, confirmed by volume and trend alignment.
Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.
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
    
    # Get 12h data for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper band = max(high, lookback=20)
    # Lower band = min(low, lookback=20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA34 for trend filter
    close_series = pd.Series(close_12h)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume 20-period average for confirmation
    vol_series = pd.Series(volume_12h)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume_12h[i // 48] > 1.5 * vol_ma_20_aligned[i] if i // 48 < len(volume_12h) else False
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper with volume and uptrend (price > EMA34)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower with volume and downtrend (price < EMA34)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 12h Donchian lower or trend reverses
            if (close[i] < donchian_lower_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 12h Donchian upper or trend reverses
            if (close[i] > donchian_upper_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hDonchian20_Breakout_Volume_EMA34_Trend"
timeframe = "4h"
leverage = 1.0