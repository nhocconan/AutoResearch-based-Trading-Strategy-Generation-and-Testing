#!/usr/bin/env python3
name = "1h_4hDonchian_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_20_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    upper_4h = align_htf_to_ltf(prices, df_4h, high_20_4h)
    lower_4h = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h volume spike detection (24-period average = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 4h upper Donchian with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > upper_4h[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: break below 4h lower Donchian with volume and daily downtrend
            elif close[i] < lower_4h[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below 4h lower Donchian or volume drops
            if close[i] < lower_4h[i] or volume[i] < vol_ma_24[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above 4h upper Donchian or volume drops
            if close[i] > upper_4h[i] or volume[i] < vol_ma_24[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Donchian breakout with 4h structure and 1d trend filter
# - Uses 4h Donchian channels (20-period) for structural support/resistance
# - Enters on 1h breakouts with volume confirmation (1.5x average volume)
# - Trend filter: daily EMA(50) slope ensures alignment with higher timeframe trend
# - Exits when price returns to opposite Donchian band or volume weakens
# - Position size 0.20 limits risk while allowing meaningful participation
# - Designed for low frequency: targets 15-30 trades/year to avoid fee drag
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Volume confirmation reduces false breakouts during low participation periods
# - Multi-timeframe alignment: 4b structure + 1d trend + 1h execution timing