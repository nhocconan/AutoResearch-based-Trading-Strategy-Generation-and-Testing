#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Donchian(20) breakout as primary signal, filtered by 1w EMA50 trend and volume confirmation.
Long when price breaks above 1w Donchian upper with volume > 1.5x 20-period average and close > 1w EMA50.
Short when price breaks below 1w Donchian lower with volume > 1.5x 20-period average and close < 1w EMA50.
Exit when price crosses the 1w EMA50 in the opposite direction.
Designed to capture medium-term trends with institutional reference points (weekly structure) while avoiding false breakouts in choppy markets.
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
    
    # Get 1w data for Donchian channels and EMA50
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian(20) channels
    # Upper = max(high_1w over last 20 periods)
    # Lower = min(low_1w over last 20 periods)
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    close_s = pd.Series(close_1w)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume 20-period average for confirmation
    volume_s = pd.Series(volume_1w)
    vol_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period average
        volume_confirmed = volume_1w[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper with volume and uptrend (close > EMA50)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with volume and downtrend (close < EMA50)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below 1w EMA50 (trend reversal)
            if close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above 1w EMA50 (trend reversal)
            if close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wDonchian20_Volume_EMA50_Trend"
timeframe = "1d"
leverage = 1.0