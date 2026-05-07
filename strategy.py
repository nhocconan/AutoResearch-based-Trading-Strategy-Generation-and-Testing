#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
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
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    R1_1d = close_1d + range_1d * 1.0833
    S1_1d = close_1d - range_1d * 1.0833
    
    # Align Camarilla levels to 1h
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume spike detection (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        trend_up = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
        trend_down = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]
        
        if position == 0:
            # Long: break above R1 with volume and 4h uptrend
            if close[i] > R1_1d_aligned[i] and vol_condition and trend_up:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume and 4h downtrend
            elif close[i] < S1_1d_aligned[i] and vol_condition and trend_down:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S1 or trend reversal
            if close[i] < S1_1d_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R1 or trend reversal
            if close[i] > R1_1d_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter
# - Uses 1d Camarilla pivot levels (R1/S1) for key intraday support/resistance
# - Enters long on break above R1 with volume spike and 4h uptrend
# - Enters short on break below S1 with volume spike and 4h downtrend
# - 4h EMA50 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Session filter (08-20 UTC) avoids low-liquidity periods
# - Exits when price returns to S1 (for longs) or R1 (for shorts) or trend reverses
# - Position size 0.20 balances risk and return
# - Targets 20-40 trades/year to stay within limits and minimize fee drag
# - Works in bull markets (buying R1 breaks in uptrend) and bear markets (selling S1 breaks in downtrend)