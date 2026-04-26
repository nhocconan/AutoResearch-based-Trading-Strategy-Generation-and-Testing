#!/usr/bin/env python3
"""
1h_HTF_Camarilla_Breakout_4hTrend_VolumeRegime
Hypothesis: On 1h timeframe, enter long when price breaks above 4h Camarilla R1 AND 4h trend is up (close > EMA34) AND volume > 1.5x 20-period average. Enter short when price breaks below 4h Camarilla S1 AND 4h trend is down (close < EMA34) AND volume > 1.5x 20-period average. Uses 4h for signal direction (Camarilla levels + EMA34 trend) and 1h only for entry timing precision. Discrete sizing (0.0, ±0.20) to limit fee drag. Target: 15-35 trades/year per symbol. Works in bull via breakouts with trend, in bear via mean-reversion failsafes (exit on trend reversal). Volume regime filter avoids low-participation false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF direction and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # need at least previous bar for Camarilla and EMA
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of completed 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_raw = df_4h['close'].values  # raw 4h close for Camarilla calculation
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous completed 4h bar to avoid look-ahead
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h_raw, 1)
    
    # First bar has no previous bar, set to NaN
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_range = prev_high_4h - prev_low_4h
    r1 = prev_close_4h + 1.1 * camarilla_range / 12
    s1 = prev_close_4h - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced for trade frequency)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # 4h trend filter
        trend_uptrend = close[i] > ema_34_4h_aligned[i]
        trend_downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + 4h uptrend
            long_signal = breakout_up and volume_spike[i] and trend_uptrend
            
            # Short: breakout below S1 + volume spike + 4h downtrend
            short_signal = breakout_down and volume_spike[i] and trend_downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price falls below R1 OR trend change to downtrend
            if close[i] < r1_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price rises above S1 OR trend change to uptrend
            if close[i] > s1_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HTF_Camarilla_Breakout_4hTrend_VolumeRegime"
timeframe = "1h"
leverage = 1.0