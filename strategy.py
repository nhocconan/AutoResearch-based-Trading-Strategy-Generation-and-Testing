#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Donchian(20) high AND 1w EMA50 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Donchian(20) low AND 1w EMA50 downtrend AND volume > 1.5 * volume_ma(20)
- Uses Donchian channels from completed 12h bars for structure-based breakouts
- 1w EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike confirms institutional participation and reduces false breakouts
- Designed for low frequency (target 12-37 trades/year) to minimize fee drag on 12h timeframe
- Exit on opposite Donchian level touch or trend reversal
- Novelty: Combines Donchian breakouts with weekly trend and volume confirmation for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Donchian levels (structure)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian(20) levels from prior 12h bar (completed bar only)
    lookback = 20
    donch_high = pd.Series(df_12h['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(df_12h['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to 12h timeframe (no additional delay needed for structure)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Load 1w data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter (needs completed 1w candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(trend_1w[i]) or np.isnan(volume_ma[i])):
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
            # Long: Price breaks above Donchian high AND 1w uptrend AND volume spike
            if close[i] > donch_high_aligned[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1w downtrend AND volume spike
            elif close[i] < donch_low_aligned[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR 1w trend turns down
            if close[i] < donch_low_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR 1w trend turns up
            if close[i] > donch_high_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0