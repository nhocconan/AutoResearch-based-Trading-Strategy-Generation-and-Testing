#!/usr/bin/env python3
name = "6h_1d_1w_Chaikin_Money_Flow_Trend"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily Chaikin Money Flow (20-period)
    mfm = ((df_1d['close'] - df_1d['low']) - (df_1d['high'] - df_1d['close'])) / (df_1d['high'] - df_1d['low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)  # handle division by zero
    mfv = mfm * df_1d['volume']
    cmf_20 = (mfv.rolling(window=20, min_periods=20).sum() / 
              df_1d['volume'].rolling(window=20, min_periods=20).sum()).values
    
    # Calculate weekly trend: EMA(34) on weekly close
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily and weekly data to 6h timeframe
    cmf_20_aligned = align_htf_to_ltf(prices, df_1d, cmf_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 6h volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 4)  # Wait for CMF, weekly EMA, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(cmf_20_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF > 0.1 (buying pressure) + weekly uptrend + volume spike
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if cmf_20_aligned[i] > 0.1 and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.1 (selling pressure) + weekly downtrend + volume spike
            elif cmf_20_aligned[i] < -0.1 and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CMF turns negative or volume drops
            if cmf_20_aligned[i] < 0 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CMF turns positive or volume drops
            if cmf_20_aligned[i] > 0 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Chaikin Money Flow with weekly trend and volume confirmation
# - CMF > 0.1 indicates institutional buying pressure, < -0.1 selling pressure
# - Weekly EMA(34) filter ensures we trade with the higher timeframe trend
# - Volume spike (1.5x average) confirms institutional participation
# - Works in both bull (buy when CMF>0.1 in weekly uptrend) and bear (sell when CMF<-0.1 in weekly downtrend)
# - Exit when CMF crosses zero or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Uses actual daily and weekly Chaikin Money Flow for institutional flow analysis
# - Designed to work in BOTH bull and bear markets via weekly trend filter