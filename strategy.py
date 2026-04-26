#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Donchian(20) high AND 1d EMA34 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Donchian(20) low AND 1d EMA34 downtrend AND volume > 1.5 * volume_ma(20)
- Uses Donchian channels from completed 12h bars for structure-based breakouts
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike confirms institutional participation and reduces false breakouts
- Designed for low frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite Donchian level touch or trend reversal
- Novelty: Combines Donchian breakouts with HTF trend and volume confirmation for BTC/ETH edge in both bull/bear markets
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
    
    # Load 12h data ONCE before loop for Donchian levels (structure)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian(20) levels from prior 12h bar (completed bar only)
    # Donchian high = max(high, lookback=20), low = min(low, lookback=20)
    lookback = 20
    donch_high = pd.Series(df_12h['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(df_12h['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to 12h timeframe (no additional delay needed for structure)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter (needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for Donchian and volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i])):
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
            # Long: Price breaks above Donchian high AND 1d uptrend AND volume spike
            if close[i] > donch_high_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1d downtrend AND volume spike
            elif close[i] < donch_low_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR 1d trend turns down
            if close[i] < donch_low_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR 1d trend turns up
            if close[i] > donch_high_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0