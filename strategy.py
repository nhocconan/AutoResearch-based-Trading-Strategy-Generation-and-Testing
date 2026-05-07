# 1h_Camarilla_R3S3_Breakout_1dTrend_Volume with Session Filter
# Target: 15-37 trades/year on 1h timeframe using 1d for direction/trend
# Session filter: 08-20 UTC to avoid low-volume Asian session noise
# Uses 1d Camarilla levels and trend filter, enters on 1h breakout with volume confirmation
# Position size: 0.20 for controlled risk
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)

#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) - avoid low liquidity periods
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load daily data ONCE for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous day (standard formula)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    pivot = (c_high + c_low + c_close) / 3
    range_val = c_high - c_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align pivot levels to 1h timeframe
    r3_1h = align_htf_to_ltf(prices, df_1d, r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(c_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 24-period average on 1h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(ema_34_1h[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_24[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in daily uptrend with volume
            if close[i] > r3_1h[i] and ema_34_1h[i] > ema_34_1h[i-1] and vol_condition:
                signals[i] = 0.20
                position = 1
            # Short: break below S3 in daily downtrend with volume
            elif close[i] < s3_1h[i] and ema_34_1h[i] < ema_34_1h[i-1] and vol_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            pivot_1h = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] < pivot_1h[i] or ema_34_1h[i] < ema_34_1h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            pivot_1h = align_htf_to_ltf(prices, df_1d, pivot)
            if close[i] > pivot_1h[i] or ema_34_1h[i] > ema_34_1h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R3/S3 breakouts with daily trend filter and volume confirmation
# - Uses 1d timeframe for structure (Camarilla levels) and trend filter (EMA34)
# - 1h timeframe only for entry timing precision, reducing false signals
# - Session filter (08-20 UTC) avoids low-volume Asian session noise
# - Volume confirmation (2x average) reduces false breakouts
# - Position size 0.20 targets ~15-37 trades/year to avoid fee drag
# - Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)