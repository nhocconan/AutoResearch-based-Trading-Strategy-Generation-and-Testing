#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-filtered strategy using 4h Donchian breakout for direction and 1d volume spike for confirmation
# Works in bull markets: breakout above 4h upper band + 1d volume spike + uptrend bias (price > 4h EMA20)
# Works in bear markets: breakout below 4h lower band + 1d volume spike + downtrend bias (price < 4h EMA20)
# Session filter (08-20 UTC) reduces noise trades. Discrete sizing (0.20) controls fee drag.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years) to avoid fee drag kill zone.

name = "1h_Session_Filtered_4hDonchian_1dVolSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex from parquet
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels and EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for Donchian(20) and EMA(20)
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    highest_20_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe (waits for completed 4h bar)
    highest_20_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_20_4h)
    lowest_20_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_20_4h)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume EMA20 for spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Volume spike: current 1d volume > 2.0 * 20-period EMA
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ema_20_1d)
    # Align volume spike to 1h timeframe (waits for completed 1d candle)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any indicator is NaN (not enough data yet)
        if (np.isnan(highest_20_4h_aligned[i]) or np.isnan(lowest_20_4h_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above 4h upper Donchian band + volume spike + uptrend
            if high[i] > highest_20_4h_aligned[i-1] and volume_spike_1d_aligned[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: Breakout below 4h lower Donchian band + volume spike + downtrend
            elif low[i] < lowest_20_4h_aligned[i-1] and volume_spike_1d_aligned[i] and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below 4h lower Donchian band (reversal) OR trend changes to down
            if low[i] < lowest_20_4h_aligned[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above 4h upper Donchian band (reversal) OR trend changes to up
            if high[i] > highest_20_4h_aligned[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals