#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA(50) trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance on 1h
# Breakout above R1 or below S1 with volume confirmation captures institutional moves
# 4h EMA(50) ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA) filters low-probability breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in bull/bear markets by following 4h trend direction for entries

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from 1d data (standard formula)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    camarilla_range = (high_1d - low_1d)
    r1 = close_1d + (1.1 * camarilla_range / 12)
    s1 = close_1d - (1.1 * camarilla_range / 12)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 4h trend filter
        # Long: Break above R1 + price above 4h EMA50 + volume spike
        # Short: Break below S1 + price below 4h EMA50 + volume spike
        if position == 0:
            if close[i] > r1_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            elif close[i] < s1_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S1 (reversion to mean) OR below 4h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R1 (reversion to mean) OR above 4h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals