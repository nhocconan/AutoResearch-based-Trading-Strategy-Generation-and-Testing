#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channels provide clear trend-following breakouts with proven efficacy in crypto
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation filters false breakouts. Target: 12-37 trades/year on 12h timeframe
# Uses discrete position sizing (0.25) to balance return and drawdown control
# Works in bull markets (breakout above upper + 1d EMA50 up) and bear markets (breakout below lower + 1d EMA50 down)

name = "12h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d data for Donchian channel calculation (20-period)
    donchian_period = 20
    if len(df_1d) < donchian_period:
        return np.zeros(n)
    
    # Calculate Donchian levels from 1d data
    highest_20 = pd.Series(df_1d['high'].values).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_20 = pd.Series(df_1d['low'].values).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align Donchian levels to 12h timeframe (wait for 1d bar to close)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Volume confirmation (volume spike > 1.8 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and Donchian calculation)
    start_idx = max(50, donchian_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian upper with volume confirmation and uptrend
            if high[i] > highest_20_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian lower with volume confirmation and downtrend
            elif low[i] < lowest_20_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower (reversal) OR trend changes to downtrend
            if low[i] < lowest_20_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper (reversal) OR trend changes to uptrend
            if high[i] > highest_20_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals