#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 12h EMA trend filter and volume spike confirmation.
# Williams %R measures overbought/oversold levels: Long when %R < -80 (oversold) and turning up,
# Short when %R > -20 (overbought) and turning down. Uses 12h EMA50 for trend filter to trade
# with the higher timeframe momentum. Volume spike confirms institutional participation.
# Designed to capture mean reversals in both bull and bear markets by fading extremes
# only when aligned with the 12h trend. Targets 20-50 trades/year to minimize fee drag.

name = "4h_WilliamsR_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        williams_r_oversold = williams_r[i] < -80
        williams_r_overbought = williams_r[i] > -20
        williams_r_turning_up = i > 0 and williams_r[i] > williams_r[i-1]
        williams_r_turning_down = i > 0 and williams_r[i] < williams_r[i-1]
        
        # Volume spike condition: current 4h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 2.0)
        
        # 12h EMA50 trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Oversold AND turning up AND volume spike AND 12h uptrend AND session
            if williams_r_oversold and williams_r_turning_up and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND turning down AND volume spike AND 12h downtrend AND session
            elif williams_r_overbought and williams_r_turning_down and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Overbought OR turning down OR 12h trend turns down
            if williams_r_overbought or williams_r_turning_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Oversold OR turning up OR 12h trend turns up
            if williams_r_oversold or williams_r_turning_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals