#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    c = close
    h = high
    l = low
    R4 = c + (range_val * 1.1 / 2)
    R3 = c + (range_val * 1.1 / 4)
    R2 = c + (range_val * 1.1 / 6)
    R1 = c + (range_val * 1.1 / 12)
    S1 = c - (range_val * 1.1 / 12)
    S2 = c - (range_val * 1.1 / 6)
    S3 = c - (range_val * 1.1 / 4)
    S4 = c - (range_val * 1.1 / 2)
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    R1_d, R2_d, R3_d, R4_d, S1_d, S2_d, S3_d, S4_d = calculate_camarilla(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    R2_d_aligned = align_htf_to_ltf(prices, df_1d, R2_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    S2_d_aligned = align_htf_to_ltf(prices, df_1d, S2_d)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2x 20-period average
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above R1 with volume in daily uptrend
            if close[i] > R1_d_aligned[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in daily downtrend
            elif close[i] < S1_d_aligned[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to S1 or trend reverses
            if close[i] < S1_d_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to R1 or trend reverses
            if close[i] > R1_d_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R1/S1 breakout with daily trend filter and volume confirmation
# - Camarilla R1/S1 levels act as intraday support/resistance derived from prior day's range
# - Breakout above R1 with volume confirms bullish momentum; breakdown below S1 confirms bearish
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend (works in bull/bear)
# - Volume confirmation (2x average) reduces false breakouts during low-liquidity periods
# - Exits when price returns to opposite level (S1 for longs, R1 for shorts) or trend reverses
# - Position size 0.25 balances risk/reward while keeping trade frequency manageable
# - Targets ~15-30 trades/year to stay within 12h limits (60-120 total over 4 years)
# - Proven pattern: Camarilla breakouts + volume + trend filter performed well on ETH/SOL in DB
# - Uses discrete position sizes to minimize fee churn from small position changes
# - Avoids overtrading by requiring multiple confluence factors (level break + volume + trend)