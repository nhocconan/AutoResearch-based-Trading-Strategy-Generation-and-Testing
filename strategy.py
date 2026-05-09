#!/usr/bin/env python3
# 4h_ChaikinMoneyFlow_DonchianBreakout_Trend
# Strategy: Trade Donchian breakouts with Chaikin Money Flow confirmation and trend filter
# Long when price breaks above Donchian(20) high with CMF > 0 and price > 12h EMA(50)
# Short when price breaks below Donchian(20) low with CMF < 0 and price < 12h EMA(50)
# Exit when price reverses to opposite Donchian level or CMF crosses zero
# Uses volume-weighted accumulation/distribution to confirm breakouts and trend filter to avoid false signals
# Designed for 4h timeframe with selective entries to minimize trade frequency

name = "4h_ChaikinMoneyFlow_DonchianBreakout_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = Sum(Money Flow Volume, 20) / Sum(Volume, 20)
    
    mfm = np.zeros(n)
    mfv = np.zeros(n)
    
    for i in range(n):
        if high[i] != low[i]:
            mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
        else:
            mfm[i] = 0
        mfv[i] = mfm[i] * volume[i]
    
    cmf = np.full(n, np.nan)
    vol_sum = np.full(n, np.nan)
    
    for i in range(19, n):
        cmf[i] = np.sum(mfv[i-19:i+1]) / np.sum(volume[i-19:i+1])
    
    # Calculate 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(cmf[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high with CMF > 0 and above 12h EMA50 (uptrend)
            if close[i] > donchian_high[i] and cmf[i] > 0 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with CMF < 0 and below 12h EMA50 (downtrend)
            elif close[i] < donchian_low[i] and cmf[i] < 0 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or CMF crosses below zero
            if close[i] < donchian_low[i] or cmf[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or CMF crosses above zero
            if close[i] > donchian_high[i] or cmf[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals