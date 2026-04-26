#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Donchian(20) high AND 12h EMA50 uptrend AND volume > 1.5 * volume_ma(20)
Short when price breaks below Donchian(20) low AND 12h EMA50 downtrend AND volume > 1.5 * volume_ma(20)
Uses Donchian channels from completed 4h bars for structure-based breakouts
12h EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
Volume spike confirms institutional participation and reduces false breakouts
Designed for moderate frequency (target 19-50 trades/year) to minimize fee drag
Exit on opposite Donchian level touch or trend reversal
Novelty: Combines Donchian breakouts with HTF trend and volume confirmation for BTC/ETH edge in both bull/bear markets
Added: Increased volume threshold to 2.0 to reduce trade frequency and avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian levels (structure)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Donchian(20) levels from prior 4h bar (completed bar only)
    # Donchian high = max(high, lookback=20), low = min(low, lookback=20)
    lookback = 20
    donch_high = pd.Series(df_4h['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(df_4h['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to 4h timeframe (no additional delay needed for structure)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Load 12h data ONCE before loop for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter (needs completed 12h candle)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_12h = np.where(ema_50_12h_aligned > 0, 
                         np.where(close > ema_50_12h_aligned, 1, -1), 
                         0)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation (increased from 1.5)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(trend_12h[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Donchian high AND 12h uptrend AND volume spike
            if close[i] > donch_high_aligned[i] and trend_12h[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 12h downtrend AND volume spike
            elif close[i] < donch_low_aligned[i] and trend_12h[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR 12h trend turns down
            if close[i] < donch_low_aligned[i] or trend_12h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR 12h trend turns up
            if close[i] > donch_high_aligned[i] or trend_12h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0