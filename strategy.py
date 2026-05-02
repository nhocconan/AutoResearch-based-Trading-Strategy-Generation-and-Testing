#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA(34) trend filter and volume confirmation
# Uses 1h primary timeframe with 4h HTF for trend alignment and 1d HTF for session filter.
# Breakouts at Camarilla R1 (resistance 1) or S1 (support 1) levels in direction of 4h EMA(34)
# with volume confirmation capture institutional moves. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (15-37/year) to minimize fee drag in 1h timeframe.
# Works in both bull and bear markets by following the 4h trend direction only.

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA(34) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA to 1h timeframe (wait for completed 4h bar)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for Camarilla calculation (previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 levels based on previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_range_1d = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + 1.05 * camarilla_range_1d  # Camarilla R1
    s1_1d = prev_close_1d - 1.05 * camarilla_range_1d  # Camarilla S1
    
    # Align 1d Camarilla levels to 1h timeframe (wait for completed 1d bar)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation (1.8x 24-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 35  # max(34 for EMA, 24 for volume MA +1 for shift, 1 for Camarilla shift)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R1 + above 4h EMA(34) + volume spike
            if (close[i] > r1_1d_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S1 + below 4h EMA(34) + volume spike
            elif (close[i] < s1_1d_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below Camarilla S1 (mean reversion) or below 4h EMA(34) (trend reversal)
            if close[i] < s1_1d_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns above Camarilla R1 (mean reversion) or above 4h EMA(34) (trend reversal)
            if close[i] > r1_1d_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals