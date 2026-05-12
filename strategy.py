#!/usr/bin/env python3
# 4h_Chaikin_Money_Flow_Trend_Pullback
# Hypothesis: Use Chaikin Money Flow (CMF) with daily trend filter and pullback entry.
# Long when CMF > 0.25, price pulls back to EMA21, and price > daily EMA50.
# Short when CMF < -0.25, price pulls back to EMA21, and price < daily EMA50.
# Exit when CMF crosses back toward zero or price breaks EMA50 in opposite direction.
# Designed to capture trend continuation after pullbacks with institutional flow confirmation.
# Works in bull/bear markets via daily trend filter. Targets ~25-35 trades/year.

name = "4h_Chaikin_Money_Flow_Trend_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    
    # CMF = 20-period sum of MFV / 20-period sum of volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.divide(mfv_sum, vol_sum, out=np.zeros_like(mfv_sum), where=vol_sum!=0)
    
    # EMA21 for pullback entry
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(cmf[i]) or np.isnan(ema21[i]) or 
            np.isnan(daily_ema_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: CMF > 0.25, price pulls back to EMA21 (close <= EMA21), and price > daily EMA
            if cmf[i] > 0.25 and close[i] <= ema21[i] and close[i] > daily_ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.25, price pulls back to EMA21 (close >= EMA21), and price < daily EMA
            elif cmf[i] < -0.25 and close[i] >= ema21[i] and close[i] < daily_ema_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF drops below 0 or price breaks below daily EMA
            if cmf[i] < 0.0 or close[i] < daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF rises above 0 or price breaks above daily EMA
            if cmf[i] > 0.0 or close[i] > daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals