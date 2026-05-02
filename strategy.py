#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 4h trend filter and 1d volume confirmation
# Uses 1h primary timeframe targeting 15-37 trades/year (60-150 total over 4 years)
# 4h EMA50 provides intermediate-term trend filter to avoid counter-trend entries
# 1d volume spike confirms institutional participation during breakouts
# 1h Donchian(20) breakout provides precise entry timing in direction of higher timeframe trend
# Works in bull (breakouts above upper band with uptrend filter) and bear (breakdowns below lower band with downtrend filter)
# Discrete position sizing (0.20) minimizes fee churn while maintaining adequate exposure
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Designed to avoid overtrading by requiring confluence of trend, volume, and breakout

name = "1h_Donchian_4hTrend_1dVol_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume EMA20 for spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # 1h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50
        bullish_bias = close[i] > ema_50_4h_aligned[i]
        bearish_bias = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation: current 1h volume > 1.5 * 1d average volume
        volume_spike = volume[i] > 1.5 * vol_ema_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and volume_spike:
                # Long: price breaks above Donchian upper channel
                if high[i] > highest_high[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias and volume_spike:
                # Short: price breaks below Donchian lower channel
                if low[i] < lowest_low[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid entries without confluence
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower channel or trend changes
            if low[i] < lowest_low[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper channel or trend changes
            if high[i] > highest_high[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals