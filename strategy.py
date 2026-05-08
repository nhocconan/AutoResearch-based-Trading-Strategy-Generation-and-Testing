#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily trend filter: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Daily volume confirmation: current volume > 1.5 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma20d * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 12h Donchian channels (20-period)
    donch_len = 20
    high_roll = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    low_roll = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donch_len  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Donchian with daily uptrend and volume spike
            long_cond = (close[i] > high_roll[i-1] and trend_1d_aligned[i] > 0.5 and vol_spike_aligned[i])
            
            # Short entry: price breaks below lower Donchian with daily downtrend and volume spike
            short_cond = (close[i] < low_roll[i-1] and trend_1d_aligned[i] < 0.5 and vol_spike_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian (trend reversal)
            if close[i] < low_roll[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian (trend reversal)
            if close[i] > high_roll[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian breakout with daily trend filter and volume confirmation.
# Works in bull markets (trend continuation) and bear markets (trend reversals).
# Daily EMA50 ensures alignment with longer-term trend, reducing counter-trend trades.
# Volume spike filter (1.5x 20-day average) ensures momentum confirmation.
# Target: 20-40 trades/year to minimize fee decay while capturing significant moves.