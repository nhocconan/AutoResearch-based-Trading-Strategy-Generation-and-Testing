#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and 1h EMA trend filter
# Long when: price touches Camarilla S1 (support) on 4h, 1d volume > 2x 20-period average, 1h EMA(20) rising
# Short when: price touches Camarilla R1 (resistance) on 4h, 1d volume > 2x 20-period average, 1h EMA(20) falling
# Exit when: price reaches Camarilla C (midpoint) or trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 20-40 trades/year.
# Uses Camarilla levels for mean reversion in ranging markets and trend alignment for momentum.
# Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets.

name = "4h_Camarilla_S1R1_1dVolSpike_1hEMA"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Camarilla levels for 4H (based on previous day's OHLC)
    # We'll use daily OHLC to calculate Camarilla levels for intraday periods
    # For 4H timeframe, we calculate based on previous daily candle
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC
    # Camarilla formulas:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # H2 = Close + 1.166 * (High - Low)
    # H1 = Close + 1.0833 * (High - Low)
    # L1 = Close - 1.0833 * (High - Low)
    # L2 = Close - 1.166 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Where High, Low, Close are from previous day
    
    # We need to align daily data to 4H intervals
    # For each 4H bar, we use the previous day's OHLC to calculate levels
    
    # Get previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_day_close = df_1d['close'].shift(1).values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    hl_range = prev_day_high - prev_day_low
    r1 = prev_day_close + 1.0833 * hl_level  # H1
    s1 = prev_day_close - 1.0833 * hl_level  # L1
    c = prev_day_close  # Camarilla pivot point (Close)
    
    # Align Camarilla levels to 4H timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    c_aligned = align_htf_to_ltf(prices, df_1d, c)
    
    # Get 1h data for EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # 1h EMA(20) for trend filter
    close_1h = df_1h['close']
    ema_20_1h = close_1h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_prev = np.roll(ema_20_1h, 1)
    ema_20_1h_prev[0] = ema_20_1h[0]
    ema_rising = ema_20_1h > ema_20_1h_prev
    ema_falling = ema_20_1h < ema_20_1h_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1h, ema_falling)
    
    # Volume spike: 1d volume > 2x 20-period average
    vol_ma_1d = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(c_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches S1 + 1d volume spike + 1h EMA rising
            if (low[i] <= s1_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                ema_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 + 1d volume spike + 1h EMA falling
            elif (high[i] >= r1_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  ema_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches C (midpoint) OR trend turns down
            if (high[i] >= c_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches C (midpoint) OR trend turns up
            if (low[i] <= c_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals