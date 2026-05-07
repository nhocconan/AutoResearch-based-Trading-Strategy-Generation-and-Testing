#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Load 1-day data ONCE for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align pivot levels to 12h timeframe (available after daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA34 and volume MA are ready
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 in daily uptrend with volume spike
            if close[i] > r3_aligned[i] and ema_34_aligned[i] > ema_34_aligned[i-1] and volume[i] > vol_ma_20[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 in daily downtrend with volume spike
            elif close[i] < s3_aligned[i] and ema_34_aligned[i] < ema_34_aligned[i-1] and volume[i] > vol_ma_20[i] * 2.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns below S3 or trend reverses
            if close[i] < s3_aligned[i] or ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns above R3 or trend reverses
            if close[i] > r3_aligned[i] or ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# - Uses Camarilla pivot levels (R3/S3) from previous day as key support/resistance
# - Long when price breaks above R3 in daily uptrend (EMA34 rising) with volume confirmation
# - Short when price breaks below S3 in daily downtrend (EMA34 falling) with volume confirmation
# - Exits when price returns to opposite level (S3 for longs, R3 for shorts) or trend reverses
# - Volume confirmation (2x average) reduces false breakouts
# - Position size 0.25 targets ~20-50 trades/year to stay within limits
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Daily trend filter ensures alignment with higher timeframe trend
# - Proven pattern: Similar to top performers (e.g., 4H_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Dyn)
# - Expected trades: ~40-80 total over 4 years (10-20/year) to avoid fee drag
# - Camarilla levels provide mathematical structure based on previous day's range
# - Daily EMA34 trend filter reduces whipsaws vs same-timeframe signals
# - Novel combination: Camarilla R3/S3 + daily trend + volume spike on 12h timeframe