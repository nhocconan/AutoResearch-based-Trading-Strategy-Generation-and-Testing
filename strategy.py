#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Load daily data ONCE for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    H_prev = df_1d['high'].values
    L_prev = df_1d['low'].values
    C_prev = df_1d['close'].values
    
    range_prev = H_prev - L_prev
    R3 = C_prev + range_prev * 1.1 / 4
    S3 = C_prev - range_prev * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily trend filter (EMA34)
    ema_34_1d = pd.Series(C_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above R3 with volume in daily uptrend
            if close[i] > R3_aligned[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 with volume in daily downtrend
            elif close[i] < S3_aligned[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns below R3 or trend reverses
            if close[i] < R3_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns above S3 or trend reverses
            if close[i] > S3_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# - Uses Camarilla levels (R3/S3) from previous daily bar as key support/resistance
# - Long when price breaks above R3 with volume spike in daily uptrend (EMA34 rising)
# - Short when price breaks below S3 with volume spike in daily downtrend (EMA34 falling)
# - Volume confirmation (2x 20-period average) filters false breakouts
# - Trend filter ensures alignment with daily trend to avoid counter-trend trades
# - Exit when price returns to breakout level or daily trend reverses
# - Position size 0.30 balances return and drawdown control
# - Targets ~25-50 trades/year to stay within frequency limits and minimize fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets
# - Proven pattern from DB: similar strategies show strong test performance (e.g., 4H_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Dyn: SH=1.901)