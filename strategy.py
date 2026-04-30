#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. In bull markets: buy when bear power improves (less negative) + price > EMA13.
# In bear markets: sell when bull power deteriorates (less positive) + price < EMA13.
# Uses 1d EMA50 for higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Designed for low trade frequency (~15-30/year) to minimize fee drag.
# Works in both bull and bear by adapting to the prevailing trend via EMA50 filter.

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Trend filter from 1d EMA50
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long when bear power improves (increases) and volume confirms
                # Bear power improving means it's becoming less negative (increasing)
                if i > start_idx and curr_bear_power > bear_power[i-1] and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend:
                # In downtrend: look for short when bull power deteriorates (decreases) and volume confirms
                # Bull power deteriorating means it's becoming less positive (decreasing)
                if i > start_idx and curr_bull_power < bull_power[i-1] and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when bear power deteriorates or trend changes
            # Exit long if bear power deteriorates (decreases) or price breaks below 1d EMA50
            if curr_bear_power < bear_power[i-1] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when bull power improves or trend changes
            # Exit short if bull power improves (increases) or price breaks above 1d EMA50
            if curr_bull_power > bull_power[i-1] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals