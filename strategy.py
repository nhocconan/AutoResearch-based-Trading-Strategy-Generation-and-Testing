#!/usr/bin/env python3
name = "12h_1D_Donchian_Breakout_Volume_Trend"
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
    
    # Load 1d data ONCE for trend, channel, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian channel (20 periods) on 1d
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average (20 periods) for confirmation
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        vol_condition = volume[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        if position == 0:
            # Long: break above 1d Donchian high in uptrend (close > EMA34)
            if close[i] > donchian_high_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Donchian low in downtrend (close < EMA34)
            elif close[i] < donchian_low_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price retouches 1d Donchian low or trend reverses
            if close[i] <= donchian_low_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price retouches 1d Donchian high or trend reverses
            if close[i] >= donchian_high_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# - Uses 1d Donchian channels (20-period high/low) as structural support/resistance
# - Enters long when 12h price breaks above 1d Donchian high in 1d uptrend (EMA34 rising)
# - Enters short when 12h price breaks below 1d Donchian low in 1d downtrend (EMA34 falling)
# - Requires volume confirmation (1.5x 1d average volume) to filter false breakouts
# - Exits when price returns to opposite Donchian band or trend reverses
# - Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)
# - Position size 0.25 balances return and risk (max 25% drawdown per position)
# - Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag
# - 1d timeframe for signals reduces noise vs lower timeframes
# - Volume confirmation and trend filter increase signal quality
# - Donchian breakouts capture momentum after consolidation periods
# - Simple, robust logic with minimal overfitting risk