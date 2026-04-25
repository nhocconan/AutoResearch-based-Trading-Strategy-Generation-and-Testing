#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above upper Donchian channel in uptrend (close > 12h EMA50) with volume spike (>2.0x 20-bar average).
Short when price breaks below lower Donchian channel in downtrend (close < 12h EMA50) with volume spike.
Exit when price re-enters Donchian channel or trend reverses.
Designed for 20-40 trades/year on 4h timeframe with tight entry conditions to minimize fee drag.
Works in bull markets via trend-following breakouts and in bear markets via counter-trend fades on extreme volume spikes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_12h_aligned[i]
        upper_channel = high_ma_20[i]
        lower_channel = low_ma_20[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (12h)
                # Long: break above upper Donchian with volume spike
                long_signal = (close[i] > upper_channel) and vol_spike[i]
                # Short: break below lower Donchian only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < lower_channel) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (12h)
                # Short: break below lower Donchian with volume spike
                short_signal = (close[i] < lower_channel) and vol_spike[i]
                # Long: break above upper Donchian only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > upper_channel) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
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
            exit_signal = (close[i] < upper_channel and close[i] > lower_channel) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter Donchian channel or trend reversal
            exit_signal = (close[i] > lower_channel and close[i] < upper_channel) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0