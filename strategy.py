#!/usr/bin/env python3
name = "4h_Donchian20_1dTrend_VolumeSqueeze_v1"
timeframe = "4h"
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
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel (20 periods) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume squeeze: current volume < 50% of 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_squeeze = volume < (vol_ma_20 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout + daily uptrend + volume squeeze (pre-breakout consolidation)
            if (close[i] > donchian_high[i-1] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_squeeze[i-1]):  # squeeze confirmed on previous bar
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + daily downtrend + volume squeeze
            elif (close[i] < donchian_low[i-1] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_squeeze[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Donchian reversal or trend change
            if close[i] < donchian_low[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Donchian reversal or trend change
            if close[i] > donchian_high[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout with daily trend filter and volume squeeze precondition
# - Volume squeeze (low volatility) precedes breakouts, increasing success rate
# - Breakout above/below Donchian(20) with daily EMA(34) trend alignment
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Volume squeeze filter reduces false breakouts by requiring pre-breakout consolidation
# - Exits on Donchian reversal or trend change to capture full moves
# - Position size 0.25 targets 20-40 trades/year to minimize fee drag
# - Daily EMA(34) ensures alignment with higher timeframe trend
# - Effective in both trending and ranging markets due to volatility-based entry filter