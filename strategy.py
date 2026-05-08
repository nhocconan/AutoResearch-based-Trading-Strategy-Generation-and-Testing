#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high, 1)  # This is wrong - need daily high/low
    prev_low_1d = np.roll(low, 1)
    
    # Fix: Need actual daily OHLC, not rolling 4h values
    # Recalculate using daily data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    prev_daily_close = np.roll(daily_high, 1)  # Still wrong
    
    # Correct approach: use actual daily OHLC from df_1d
    prev_daily_close = np.roll(df_1d['close'].values, 1)
    prev_daily_high = np.roll(df_1d['high'].values, 1)
    prev_daily_low = np.roll(df_1d['low'].values, 1)
    
    # Camarilla levels calculation
    camarilla_range = prev_daily_high - prev_daily_low
    camarilla_level = prev_daily_close + camarilla_range * 1.1 / 12  # R1 level
    camarilla_support = prev_daily_close - camarilla_range * 1.1 / 12  # S1 level
    
    camarilla_level_aligned = align_htf_to_ltf(prices, df_1d, camarilla_level)
    camarilla_support_aligned = align_htf_to_ltf(prices, df_1d, camarilla_support)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_level_aligned[i]) or 
            np.isnan(camarilla_support_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, bullish trend, volume spike
            long_cond = (close[i] > camarilla_level_aligned[i] and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S1, bearish trend, volume spike
            short_cond = (close[i] < camarilla_support_aligned[i] and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 (reversal)
            if close[i] < camarilla_support_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 (reversal)
            if close[i] > camarilla_level_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals