#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1w trend filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1w EMA34 trend filter ensures we only trade with the weekly trend (avoid counter-trend whipsaws)
# Volume confirmation requires 6h volume > 1.5x 1d average volume to filter low-conviction moves
# Works in bull/bear: weekly trend filter adapts to market regime, Elder Ray captures momentum exhaustion
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1w_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_s_1w = pd.Series(close_1w)
    ema34_1w = close_s_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 6h timeframe (wait for 1w bar close)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d average volume (20-period) for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d average volume to 6h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Weekly trend filter: price above/below 1w EMA34
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative (momentum exhaustion) OR weekly trend turns down
            if bear_power[i] < 0 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive (momentum exhaustion) OR weekly trend turns up
            if bull_power[i] > 0 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: trade with weekly trend, confirmed by Elder Ray and volume
            if uptrend and volume_confirmed:
                # Long when Bull Power is positive and rising (bullish momentum)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif downtrend and volume_confirmed:
                # Short when Bear Power is negative and falling (bearish momentum)
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals