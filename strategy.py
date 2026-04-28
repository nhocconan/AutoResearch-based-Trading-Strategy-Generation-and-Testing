#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Donchian upper AND 12h EMA50 rising AND volume > 2x 20-bar avg
# Short when price breaks below Donchian lower AND 12h EMA50 falling AND volume > 2x 20-bar avg
# Exit when price crosses Donchian midpoint or volume drops below average
# Target: 20-50 trades/year via tight entry conditions reducing fee drag
# Works in bull markets via breakouts and bear markets via breakdowns with trend filter

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian(20) to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_spike[i]
        ema_trend = ema_50_12h_aligned[i]
        prev_ema_trend = ema_50_12h_aligned[i-1] if i > 0 else ema_trend
        ema_rising = ema_trend > prev_ema_trend
        ema_falling = ema_trend < prev_ema_trend
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND EMA50 rising AND volume spike
            if close[i] > highest_20[i] and ema_rising and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND EMA50 falling AND volume spike
            elif close[i] < lowest_20[i] and ema_falling and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses midpoint or volume drops
            if close[i] < midpoint_20[i] or not volume[i] > volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses midpoint or volume drops
            if close[i] > midpoint_20[i] or not volume[i] > volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals