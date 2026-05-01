#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves
# 12h EMA50 ensures we trade only in the direction of the higher timeframe trend
# Volume spike confirms institutional participation
# Designed for low frequency (75-200 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + breakout structure

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Donchian(20) channels - using prior 20 periods to avoid look-ahead
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need 12h EMA50 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > high_ma_20[i]  # Price breaks above upper Donchian
        breakout_short = close[i] < low_ma_20[i]  # Price breaks below lower Donchian
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian with volume spike and uptrend
            if breakout_long and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian with volume spike and downtrend
            elif breakout_short and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below lower Donchian or trend reversal
            if close[i] < low_ma_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above upper Donchian or trend reversal
            if close[i] > high_ma_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals