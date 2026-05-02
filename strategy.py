#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian channel breakout captures institutional accumulation/distribution zones
# 12h EMA50 ensures alignment with higher timeframe trend to avoid range-bound whipsaws
# Volume confirmation (2x 20-period EMA) filters low-conviction moves
# Designed for 4h timeframe targeting 19-50 trades/year (75-200 total over 4 years)
# Uses discrete position sizing (0.30) to balance return potential and drawdown control
# Works in bull markets (breakout above upper channel + 12h EMA up-trend) and bear markets (breakout below lower channel + 12h EMA down-trend)

name = "4h_Donchian20_12hEMA50_Trend_Volume"
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
    
    # 12h data for trend filter (EMA50) and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 12h EMA50 calculation
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h Donchian(20) channels
    high_20_12h = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian with volume confirmation and uptrend
            if close[i] > upper_12h_aligned[i] and uptrend and volume_confirmation[i]:
                signals[i] = 0.30
                position = 1
            # Short: Breakout below lower Donchian with volume confirmation and downtrend
            elif close[i] < lower_12h_aligned[i] and downtrend and volume_confirmation[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian (reversal) OR trend changes to downtrend
            if close[i] < lower_12h_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian (reversal) OR trend changes to uptrend
            if close[i] > upper_12h_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals