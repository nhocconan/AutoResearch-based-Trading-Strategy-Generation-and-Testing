#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation (>1.5x 20-bar volume MA) and 1w EMA50 trend filter
# Donchian channels provide robust trend-following structure. Volume spike confirms institutional participation.
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend whipsaws. Works in bull (strong breakouts with volume)
# and bear (failed reversals at key levels reverse quickly). Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_Breakout_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) on 1w close
    ema_1w_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Donchian channels (20-period) - using prior bar to avoid look-ahead
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().shift(1).values  # Prior bar's 20-period high
    donchian_low = low_roll.min().shift(1).values    # Prior bar's 20-period low
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 1w EMA and 20 for Donchian/volume MA, plus 1 for shift
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions (using prior bar levels)
        breakout_up = curr_close > donchian_high[i]   # Break above prior 20-period high
        breakout_down = curr_close < donchian_low[i]  # Break below prior 20-period low
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, price above 1w EMA50, volume spike
            if breakout_up and curr_close > ema_1w_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, price below 1w EMA50, volume spike
            elif breakout_down and curr_close < ema_1w_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or price below 1w EMA50
            if curr_close < donchian_low[i] or curr_close < ema_1w_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or price above 1w EMA50
            if curr_close > donchian_high[i] or curr_close > ema_1w_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals