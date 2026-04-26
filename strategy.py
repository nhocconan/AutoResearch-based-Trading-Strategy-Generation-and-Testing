#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: On 1h timeframe, enter long when price breaks above Camarilla R1 level AND 4h trend is up (close > EMA20) AND volume > 2x 20-period average volume. Enter short when price breaks below Camarilla S1 level AND 4h trend is down (close < EMA20) AND volume > 2x 20-period average volume. Uses discrete sizing (0.0, ±0.20) to limit fee drag. Camarilla R1/S1 levels provide intraday support/resistance with fewer false breakouts than R3/S3. Volume spike filter confirms participation. 4h EMA20 trend filter ensures alignment with higher timeframe momentum. Designed to generate ~15-35 trades per year on BTC/ETH/SOL with Sharpe > 0 in both bull and bear regimes by avoiding whipsaws in ranging markets.
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
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # need at least previous 4h bar for Camarilla and EMA
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of completed 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_raw = df_4h['close'].values  # raw 4h close for Camarilla calculation
    
    # Camarilla levels: based on previous 4h bar's range
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous completed 4h bar to avoid look-ahead
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h_raw, 1)
    
    # First bar has no previous 4h bar, set to NaN
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_range = prev_high_4h - prev_low_4h
    r1 = prev_close_4h + 1.1 * camarilla_range / 12
    s1 = prev_close_4h - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: volume > 2x 20-period average (moderate to balance trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(20, 20)  # EMA20 needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter: only trade during active hours
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # 4h trend filter
        trend_uptrend = close[i] > ema_20_4h_aligned[i]
        trend_downtrend = close[i] < ema_20_4h_aligned[i]
        
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0