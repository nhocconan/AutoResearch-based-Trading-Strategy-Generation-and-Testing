#!/usr/bin/env python3
name = "12h_DonchianBreakout_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) on 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # Wait for Donchian channel
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if high[i] > highest_high[i-1] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low with volume and daily downtrend
            elif low[i] < lowest_low[i-1] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price closes below Donchian low or volume drops
            if close[i] < lowest_low[i-1] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price closes above Donchian high or volume drops
            if close[i] > highest_high[i-1] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Donchian breakout captures strong momentum moves in both directions
# - Daily EMA(34) filter ensures trades align with higher timeframe trend
# - Volume spike (2x average) confirms institutional participation and reduces false breakouts
# - Works in bull markets (long breakouts in uptrend) and bear markets (short breakdowns in downtrend)
# - Exit when price returns to opposite Donchian level or volume weakens
# - Position size 0.30 balances return potential with drawdown control
# - Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# - Uses multiple timeframes: 12h for entry/exit, 1d for trend filter
# - Volume confirmation on same timeframe as entry for accuracy
# - Aims to avoid overtrading by requiring confluence of breakout, volume, and trend