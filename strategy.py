#!/usr/bin/env python3
name = "6h_1d_21EMA_Cross_4H_Donchian_Trend"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily 21 EMA for trend filter
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # 4H Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    donch_high_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # 6H volume spike detection (4-period average)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20, 4)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(donch_high_4h_aligned[i]) or 
            np.isnan(donch_low_4h_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 4H Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            daily_uptrend = ema_21_1d_aligned[i] > ema_21_1d_aligned[i-1]
            
            if close[i] > donch_high_4h_aligned[i] and vol_condition and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below 4H Donchian low with volume and daily downtrend
            elif close[i] < donch_low_4h_aligned[i] and vol_condition and not daily_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below 4H Donchian low or volume drops
            if close[i] < donch_low_4h_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above 4H Donchian high or volume drops
            if close[i] > donch_high_4h_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h 21EMA cross + 4H Donchian breakout with volume confirmation
# - Daily 21 EMA provides trend filter (works in both bull/bear markets)
# - 4H Donchian(20) breakout captures medium-term momentum
# - Volume spike (2.0x 4-period average) confirms institutional participation
# - Enter long when price breaks above 4H Donchian high with volume in daily uptrend
# - Enter short when price breaks below 4H Donchian low with volume in daily downtrend
# - Exit when price returns to opposite Donchian level or volume weakens
# - Position size 0.25 targets ~50-100 trades over 4 years (~12-25/year)
# - Uses actual 4H and daily data via mtf_data to avoid look-ahead
# - Designed to work in BOTH bull and bear markets via daily trend filter
# - Volume confirmation reduces false breakouts while maintaining edge in BTC/ETH