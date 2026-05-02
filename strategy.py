#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves
# 1d EMA50 filters for higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0 x 30-period EMA) confirms breakout validity
# Discrete position sizing (0.25) balances opportunity with fee drag control
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns
# Works in both bull and bear markets by following HTF trend direction

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
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
    
    # Volume confirmation (volume spike > 2.0 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_30)
    
    # 1d data for Donchian channel calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous 1d bar (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    lookback = 20
    if len(df_1d) < lookback:
        return np.zeros(n)
        
    donchian_upper = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian upper with volume confirmation and uptrend
            if close[i] > donchian_upper_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower with volume confirmation and downtrend
            elif close[i] < donchian_lower_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian lower (reversal to downside) OR trend changes to downtrend
            if close[i] < donchian_lower_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian upper (reversal to upside) OR trend changes to uptrend
            if close[i] > donchian_upper_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals