#!/usr/bin/env python3
name = "6h_WilliamsVixFix_Volume_Momentum"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop for VixFix
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Williams VixFix: measures volatility sentiment (high = fear)
    # Formula: ((Highest Close in period - Low) / Highest Close in period) * 100
    highest_close_22d = pd.Series(df_1d['close']).rolling(window=22, min_periods=22).max().values
    vixfix = ((highest_close_22d - df_1d['low'].values) / highest_close_22d) * 100
    
    # Align VixFix to 6h timeframe
    vixfix_aligned = align_htf_to_ltf(prices, df_1d, vixfix)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 22, 4)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(vixfix_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: High fear (VixFix > 80) + volume spike + weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if vixfix_aligned[i] > 80 and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Low fear (VixFix < 20) + volume spike + weekly downtrend
            elif vixfix_aligned[i] < 20 and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Fear subsides or trend changes
            if vixfix_aligned[i] < 40 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Fear increases or trend changes
            if vixfix_aligned[i] > 60 or weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams VixFix measures market fear/greed on daily timeframe
# - High VixFix (>80) indicates extreme fear -> potential long opportunity with volume
# - Low VixFix (<20) indicates complacency/greed -> potential short opportunity
# - Weekly EMA(34) filter ensures we trade with the higher timeframe trend
# - Volume spike (2x average) confirms institutional participation in the move
# - Works in both bull (buy fear spikes in uptrend) and bear (sell complacency in downtrend)
# - VixFix is mean-reverting, providing natural exit when fear/sentiment normalizes
# - Novel application: VixFix (typically for intraday) applied to swing trading via 6H
# - Targets 50-120 total trades over 4 years (12-30/year) to avoid fee drag
# - Uses Williams VixFix (fear gauge) + weekly trend + volume confirmation
# - Aims to catch panic selling and euphoric tops with proper trend alignment