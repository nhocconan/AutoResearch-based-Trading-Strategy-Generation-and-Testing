#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise support/resistance, 4h EMA50 filters for higher timeframe trend alignment,
# volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
# Designed for 15-37 trades/year on 1h timeframe to minimize fee drag while capturing trends in both bull and bear markets.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 trend filter from prior completed 4h bar
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_shifted = np.roll(ema50_4h, 1)
    ema50_4h_shifted[0] = np.nan
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_shifted)
    
    # Get 1d data for Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior completed 1d bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    camarilla_R1 = close_1d + 1.1 * rng / 12
    camarilla_S1 = close_1d - 1.1 * rng / 12
    
    # Shift to use only prior completed 1d bar
    camarilla_R1_shifted = np.roll(camarilla_R1, 1)
    camarilla_S1_shifted = np.roll(camarilla_S1, 1)
    camarilla_R1_shifted[0] = np.nan
    camarilla_S1_shifted[0] = np.nan
    
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_shifted)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Camarilla R1 AND 4h EMA50 uptrend AND volume spike
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: break below Camarilla S1 AND 4h EMA50 downtrend AND volume spike
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S1 OR below 4h EMA50
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Camarilla R1 OR above 4h EMA50
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals