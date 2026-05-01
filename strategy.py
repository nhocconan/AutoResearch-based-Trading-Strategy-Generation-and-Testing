#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. Weekly EMA50 ensures we only trade
# in the direction of the higher timeframe trend (long above, short below). Volume confirmation
# filters out false breakouts. Works in bull (trend continuation) and bear (trend reversals after volatility expansion).

name = "12h_Donchian20_Breakout_WeeklyEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 (using prior week to avoid look-ahead)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on 12h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > highest_high_20[i-1]  # Use previous bar's high to avoid look-ahead
        breakout_down = curr_close < lowest_low_20[i-1]   # Use previous bar's low to avoid look-ahead
        
        # Weekly EMA50 trend filter
        above_weekly_ema = curr_close > ema_50_1w_aligned[i]
        below_weekly_ema = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian upper, volume spike, above weekly EMA50
            if breakout_up and vol_spike and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower, volume spike, below weekly EMA50
            elif breakout_down and vol_spike and below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below Donchian lower or weekly EMA50 failure
            if curr_close < lowest_low_20[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above Donchian upper or weekly EMA50 failure
            if curr_close > highest_high_20[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals