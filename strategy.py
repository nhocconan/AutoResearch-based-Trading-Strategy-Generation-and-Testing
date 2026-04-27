#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper Donchian with bullish 1d trend and volume spike.
# Short when price breaks below lower Donchian with bearish 1d trend and volume spike.
# Exit when price returns to middle Donchian (mean reversion).
# Uses 1d timeframe for trend filter to reduce noise and improve win rate.
# Target: 15-25 trades/year to minimize fee dust while capturing strong moves.
# Proven pattern: Donchian + trend + volume works on SOL (see DB), extended to 12h for lower frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian Channel on 12h timeframe (20-period)
    dc_period = 20
    upper_dc = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_dc = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    middle_dc = (upper_dc + lower_dc) / 2.0
    
    # Volume filter: volume > 1.8x 20-period average (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or np.isnan(middle_dc[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian, above 1d EMA50, volume spike
        if (close[i] > upper_dc[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian, below 1d EMA50, volume spike
        elif (close[i] < lower_dc[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle Donchian (mean reversion)
        elif position == 1 and close[i] < middle_dc[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > middle_dc[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0