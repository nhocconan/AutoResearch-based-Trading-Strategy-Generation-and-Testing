# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Uses Camarilla pivot levels (R1, S1) from daily timeframe for breakout entries.
# Enters long when price breaks above R1 with volume confirmation and daily uptrend.
# Enters short when price breaks below S1 with volume confirmation and daily downtrend.
# Designed for 12h timeframe to reduce trade frequency and avoid fee drag.
# Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns.
# Target: 12-30 trades/year per symbol.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Daily high, low, close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    rang = high_1d - low_1d
    R1 = close_1d + rang * 1.1 / 12
    S1 = close_1d - rang * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume average (20-period) for confirmation
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema34_1d[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above R1 + daily uptrend + volume
            if close[i] > R1_aligned[i] and trend_1d_up_aligned[i] > 0.5 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + daily downtrend + volume
            elif close[i] < S1_aligned[i] and trend_1d_down_aligned[i] > 0.5 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns below R1 or trend changes
            if close[i] < R1_aligned[i] or trend_1d_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns above S1 or trend changes
            if close[i] > S1_aligned[i] or trend_1d_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals