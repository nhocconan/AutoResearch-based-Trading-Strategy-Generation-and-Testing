#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot (R3/S3) breakout with 1w trend filter and volume confirmation
# Camarilla levels provide high-probability reversal/breakout zones. 
# Breakout above R3 or below S3 with 1w trend alignment and volume spike indicates strong momentum.
# Works in both bull/bear by filtering breakouts with higher timeframe trend.
# Targets 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.

name = "1d_Camarilla_R3S3_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1d (using previous day's OHLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use R3 and S3 as breakout levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (already aligned as they're based on 1d)
    # But we need to shift by 1 to avoid look-ahead (use previous day's levels)
    R3_shifted = np.roll(R3, 1)
    S3_shifted = np.roll(S3, 1)
    R3_shifted[0] = np.nan
    S3_shifted[0] = np.nan
    
    # 1w trend filter: EMA(34) on weekly close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d volume spike: volume > 2x 20-day average
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike = df_1d['volume'].values > (vol_ma.values * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_shifted[i]) or np.isnan(S3_shifted[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, above 1w EMA trend, volume spike
            if close[i] > R3_shifted[i] and close[i] > ema_34_1w_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, below 1w EMA trend, volume spike
            elif close[i] < S3_shifted[i] and close[i] < ema_34_1w_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below S3 or trend weakens (price below EMA)
            if close[i] < S3_shifted[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above R3 or trend weakens (price above EMA)
            if close[i] > R3_shifted[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals