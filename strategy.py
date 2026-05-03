#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot (R1/S1) breakout + 1w EMA50 trend filter + volume confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance derived from prior day's range
# Breakouts above R1 or below S1 with volume confirmation capture institutional participation
# 1w EMA50 ensures we trade with the primary weekly trend to avoid counter-trend whipsaws
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag
# Works in bull/bear: weekly trend filter adapts to market regime while Camarilla levels dynamically adjust to volatility

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels for 1d: based on prior day's high, low, close
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's data (no look-ahead)
    high_1d_lag = np.roll(high_1d, 1)
    low_1d_lag = np.roll(low_1d, 1)
    close_1d_lag = np.roll(close_1d, 1)
    high_1d_lag[0] = np.nan
    low_1d_lag[0] = np.nan
    close_1d_lag[0] = np.nan
    
    camarilla_range = (high_1d_lag - low_1d_lag) * 1.1 / 12.0
    r1 = close_1d_lag + camarilla_range
    s1 = close_1d_lag - camarilla_range
    
    # Align Camarilla levels to 1d timeframe (already aligned as 1d data)
    r1_aligned = r1  # No additional alignment needed for same timeframe
    s1_aligned = s1
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 1w trend filter
        # Long: close breaks above R1 + price above 1w EMA50 + volume spike
        # Short: close breaks below S1 + price below 1w EMA50 + volume spike
        if position == 0:
            if (close[i] > r1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < s1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close breaks below S1 (reversion to mean) OR price below 1w EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close breaks above R1 (reversion to mean) OR price above 1w EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals