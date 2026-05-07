#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when price breaks above 1d Donchian upper (20-period high) AND 1w EMA20 uptrend AND volume > 1.5 * 20-day average.
# Short when price breaks below 1d Donchian lower (20-period low) AND 1w EMA20 downtrend AND volume > 1.5 * 20-day average.
# Uses weekly trend to avoid counter-trend trades and volume to confirm breakout strength.
# Designed for low frequency (target: 10-25 trades/year) to minimize fee drag and improve robustness.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # 1d volume average (20-period EMA)
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA20 for trend direction
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-day EMA
            vol_confirm = volume[i] > 1.5 * vol_ema_20_aligned[i]
            
            # Long condition: break above Donchian high, weekly uptrend, volume confirmation
            long_condition = (close[i] > donchian_high_aligned[i]) and \
                            (ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]) and \
                            vol_confirm
            
            # Short condition: break below Donchian low, weekly downtrend, volume confirmation
            short_condition = (close[i] < donchian_low_aligned[i]) and \
                             (ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]) and \
                             vol_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or weekly trend turns down
            if (close[i] < donchian_low_aligned[i]) or (ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or weekly trend turns up
            if (close[i] > donchian_high_aligned[i]) or (ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals