# 12h_CamarillaBreakout_1WTrend_Volume
# Hypothesis: 12h Camarilla Pivot Breakout with 1-week Trend Filter and Volume Confirmation
# - Uses weekly trend to avoid counter-trend trades
# - Daily Camarilla levels for S1/S2 (long) and R1/R2 (short)
# - Breakout requires volume spike (>2x 20-period average)
# - Target: 12-37 trades/year (50-150 total over 4 years)
# - Works in bull/bear by using weekly trend filter
# - Position size: 0.25 (25% of capital) to manage drawdown

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_CamarillaBreakout_1WTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data
    n1d = len(close_1d)
    camarilla_S1 = np.full(n1d, np.nan)
    camarilla_S2 = np.full(n1d, np.nan)
    camarilla_R1 = np.full(n1d, np.nan)
    camarilla_R2 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_val = H - L
        camarilla_S1[i] = C - range_val * 1.08
        camarilla_S2[i] = C - range_val * 1.16
        camarilla_R1[i] = C + range_val * 1.08
        camarilla_R2[i] = C + range_val * 1.16
    
    # Align Camarilla levels to 12h timeframe
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S2)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R2)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend filter (more responsive than 50)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_S2_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_R2_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S1 (support) with 1w uptrend + volume spike
            long_cond = (close[i] > camarilla_S1_aligned[i] and 
                        ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R1 (resistance) with 1w downtrend + volume spike
            short_cond = (close[i] < camarilla_R1_aligned[i] and 
                         ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S2 (strong support break)
            if close[i] < camarilla_S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R2 (strong resistance break)
            if close[i] > camarilla_R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals