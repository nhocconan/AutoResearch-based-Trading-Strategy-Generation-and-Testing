#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
# Camarilla pivot levels provide high-probability reversal/breakout zones. Break above R1
# with 12h EMA50 uptrend and volume spike signals bullish momentum. Break below S1 with
# 12h EMA50 downtrend and volume spike signals bearish momentum. Discrete sizing 0.25
# limits fee drag. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull via long breakouts and bear via short breakdowns when aligned with 12h trend.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We only need R1 and S1 for breakout signals
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    r1_1d = close_1d + (1.1 * rng / 12)
    s1_1d = close_1d - (1.1 * rng / 12)
    
    # Align 1d Camarilla levels to 4h timeframe (previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema_12h = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend
        is_uptrend = close_val > ema_12h
        is_downtrend = close_val < ema_12h
        
        # Entry logic
        if position == 0:
            # Long: Price breaks above R1 AND 12h uptrend AND volume spike
            if close_val > r1_val and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND 12h downtrend AND volume spike
            elif close_val < s1_val and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S1 OR 12h trend turns down OR volume drops
            if close_val < s1_val or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R1 OR 12h trend turns up OR volume drops
            if close_val > r1_val or is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals