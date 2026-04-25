#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume spike confirmation.
Long when price breaks above 20-day high in uptrend (close > weekly EMA50) with volume spike (>2.0x 20-day average).
Short when price breaks below 20-day low in downtrend (close < weekly EMA50) with volume spike.
Exit when price re-enters the Donchian channel or trend reverses.
Designed for 7-25 trades/year on 1d timeframe with tight entry conditions to minimize fee drag.
Works in bull markets via trend-following breakouts and in bear markets via counter-trend fades on extreme volume spikes.
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
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (weekly)
                # Long: break above 20-day high with volume spike
                long_signal = (close[i] > high_ma_20[i]) and vol_spike[i]
                # Short: break below 20-day low only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < low_ma_20[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (weekly)
                # Short: break below 20-day low with volume spike
                short_signal = (close[i] < low_ma_20[i]) and vol_spike[i]
                # Long: break above 20-day high only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > high_ma_20[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
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
            # Exit conditions: re-enter Donchian channel or trend reversal
            exit_signal = (close[i] < high_ma_20[i] and close[i] > low_ma_20[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter Donchian channel or trend reversal
            exit_signal = (close[i] > low_ma_20[i] and close[i] < high_ma_20[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0