#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel (20) breakout with 1w trend filter and volume confirmation.
Long when price breaks above 20-day high with 1w close > 1w EMA34 and volume > 1.5x 20-day average.
Short when price breaks below 20-day low with 1w close < 1w EMA34 and volume > 1.5x 20-day average.
Exit when price returns to the 20-day midpoint or reverses with volume confirmation.
Uses 1d for price action and volume, 1w for trend filter to avoid counter-trend trades.
Designed to capture strong momentum moves with institutional participation in both bull and bear markets.
Volume regime filter ensures trades occur during periods of higher conviction, reducing whipsaws.
Target: 15-25 trades/year per symbol to minimize fee drag while maintaining edge.
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
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Calculate 1d volume MA20 for regime filter
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to lower timeframe (prices index)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Align 1w EMA34 to lower timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian(20) and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        # We check if current daily volume is elevated (institutional participation)
        volume_confirmed = volume_1d[i] > 1.5 * vol_ma_20_1d[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with volume confirmation and 1w uptrend
            if (close_1d[i] > high_20_aligned[i] and 
                volume_confirmed and 
                close_1w[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume confirmation and 1w downtrend
            elif (close_1d[i] < low_20_aligned[i] and 
                  volume_confirmed and 
                  close_1w[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR breaks below low with volume (reversal)
            if (close_1d[i] <= donchian_mid_aligned[i] or 
                (close_1d[i] < low_20_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR breaks above high with volume (reversal)
            if (close_1d[i] >= donchian_mid_aligned[i] or 
                (close_1d[i] > high_20_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Regime"
timeframe = "1d"
leverage = 1.0