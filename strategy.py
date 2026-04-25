#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period high in uptrend (close > 1d EMA50) with volume spike.
Short when price breaks below 20-period low in downtrend (close < 1d EMA50) with volume spike.
Exit when price re-enters 20-period channel or trend reverses.
Designed for 20-50 trades/year on 4h timeframe to minimize fee drag and work in both bull/bear markets.
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 4h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Breakout entry logic with trend filter
            if close[i] > ema_trend:  # Uptrend regime (1d)
                # Long: break above 20-period high with volume spike
                long_signal = (close[i] > high_ma_20[i]) and vol_spike[i]
                # Short: break below 20-period low only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < low_ma_20[i]) and vol_spike[i] and (volume[i] > (3.0 * vol_ma_20[i]))
            else:  # Downtrend regime (1d)
                # Short: break below 20-period low with volume spike
                short_signal = (close[i] < low_ma_20[i]) and vol_spike[i]
                # Long: break above 20-period high only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > high_ma_20[i]) and vol_spike[i] and (volume[i] > (3.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: re-enter 20-period channel or trend reversal
            exit_signal = (close[i] < high_ma_20[i] and close[i] > low_ma_20[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter 20-period channel or trend reversal
            exit_signal = (close[i] > low_ma_20[i] and close[i] < high_ma_20[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0