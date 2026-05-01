#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian channels provide clear trend-following structure. EMA50 on 12h ensures alignment with medium-term trend.
# Volume spike confirms breakout validity. Discrete sizing (0.25) minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull (breakouts with volume) and bear (volatility expansion after consolidation).

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) calculation
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels on 4h timeframe (using prior 20 bars to avoid look-ahead)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 60  # Need 50 for EMA + 20 for Donchian + buffer
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 12h EMA50
        trend_up = curr_close > ema_50_12h_aligned[i]
        trend_down = curr_close < ema_50_12h_aligned[i]
        
        # Donchian breakout conditions (using prior bar channels to avoid look-ahead)
        breakout_up = curr_close > highest_high_20[i]  # Break above upper channel
        breakout_down = curr_close < lowest_low_20[i]  # Break below lower channel
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, volume spike, uptrend
            if breakout_up and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or trend reversal
            if curr_close < lowest_low_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or trend reversal
            if curr_close > highest_high_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals