#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: 4-hour Donchian(20) breakout with daily EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel in uptrend (close > daily EMA50) with volume spike.
Short when price breaks below lower Donchian channel in downtrend (close < daily EMA50) with volume spike.
Exit when price re-enters the Donchian channel or trend reverses.
Designed for 4h timeframe to capture medium-term trends with controlled trade frequency (target: 75-200/4 years).
Works in bull markets via breakout momentum and in bear markets via short-side breakdowns.
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
    
    # Get 4h data for Donchian channel calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels for previous 20 periods
    # Upper = MAX(high over last 20 periods)
    # Lower = MIN(low over last 20 periods)
    # We use the previous completed 4h bar's data to avoid look-ahead
    high_4h_prev = np.concatenate([[np.nan], high_4h[:-1]])
    low_4h_prev = np.concatenate([[np.nan], low_4h[:-1]])
    
    # Rolling window of 20 on previous bar data
    upper_20 = pd.Series(high_4h_prev).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h_prev).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (same timeframe, so direct use with 1-bar lag)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (daily)
                # Long: break above upper Donchian with volume spike
                long_signal = (close[i] > upper_aligned[i]) and vol_spike[i]
                # Short: break below lower Donchian only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < lower_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (daily)
                # Short: break below lower Donchian with volume spike
                short_signal = (close[i] < lower_aligned[i]) and vol_spike[i]
                # Long: break above upper Donchian only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > upper_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
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
            exit_signal = (close[i] < upper_aligned[i] and close[i] > lower_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter Donchian channel or trend reversal
            exit_signal = (close[i] > lower_aligned[i] and close[i] < upper_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0