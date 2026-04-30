#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation (2.0x 20-period average) + 1d EMA50 trend filter
# Donchian channels provide robust structure for breakouts in both bull and bear markets.
# Volume confirmation ensures institutional participation, reducing false breakouts.
# 1d EMA50 filters for higher-timeframe trend alignment to avoid counter-trend trades.
# Uses discrete position sizing 0.25 to balance profit potential and drawdown control.
# Session filter (08-20 UTC) reduces noise trades. Target: 80-180 total trades over 4 years (20-45/year).

name = "4h_Donchian20_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Calculate Donchian channels for prior 20 periods (using data up to i-1)
        donchian_high = np.max(high[max(0, i-20):i])
        donchian_low = np.min(low[max(0, i-20):i])
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above Donchian high AND above 1d EMA50 (uptrend)
                if curr_close > donchian_high and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian low AND below 1d EMA50 (downtrend)
                elif curr_close < donchian_low and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian low or below 1d EMA50
            if curr_close < donchian_low or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high or above 1d EMA50
            if curr_close > donchian_high or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals