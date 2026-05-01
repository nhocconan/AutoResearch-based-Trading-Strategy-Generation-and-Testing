#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses Donchian channel breakouts as price structure signals, filtered by 1d EMA50 trend.
# Volume confirmation ensures breakouts have sufficient market participation.
# Works in bull markets (buy upper band breakout with uptrend) and bear markets 
# (sell lower band breakdown with downtrend). Discrete position sizing (0.30) balances
# return and drawdown. Target: 75-200 trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA50 and Donchian20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Donchian breakout conditions
        breakout_upper = curr_close > high_20[i]   # Break above upper band
        breakdown_lower = curr_close < low_20[i]   # Break below lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper band AND uptrend AND volume confirmation
            if breakout_upper and uptrend and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Breakdown below lower band AND downtrend AND volume confirmation
            elif breakdown_lower and downtrend and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below lower band (reversal signal)
            if curr_close < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on breakout above upper band (reversal signal)
            if curr_close > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals