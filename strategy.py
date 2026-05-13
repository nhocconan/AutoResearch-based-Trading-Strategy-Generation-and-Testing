#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume confirmation.
# Long when price breaks above R1 with 4h EMA34 uptrend and volume > 1.5x average.
# Short when price breaks below S1 with 4h EMA34 downtrend and volume > 1.5x average.
# Uses discrete sizing 0.20. Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Camarilla R1/S1 levels provide tight institutional support/resistance with lower false breakouts.
# 4h EMA34 ensures we trade with the higher timeframe trend. Volume spike confirms participation.
# Session filter (08-20 UTC) reduces noise during low-liquidity periods.
# Designed to work in both bull (upward breaks) and bear (downward breaks) markets.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA34_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (approx using 24x 1h bars)
    lookback = 24  # 24 * 1h = 24h approx
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous "day"
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla R1 and S1 levels (tighter than R3/S3 for more precise entries)
    camarilla_range = high_prev - low_prev
    r1 = close_prev + 1.1 * camarilla_range / 4
    s1 = close_prev - 1.1 * camarilla_range / 4
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 on 4h data
    ema_34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 4h EMA34 to 1h timeframe (wait for 4h bar to close)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(avg_volume[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with 4h EMA34 uptrend and volume confirmation
            if (close[i] > r1[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with 4h EMA34 downtrend and volume confirmation
            elif (close[i] < s1[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 (reversal signal)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 (reversal signal)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals