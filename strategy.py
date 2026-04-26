#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_VolumeSpike_v2
Hypothesis: 6h Donchian(20) breakout with weekly EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Donchian(20) high AND weekly EMA50 uptrend AND volume > 2.0 * volume_ma(20)
- Short when price breaks below Donchian(20) low AND weekly EMA50 downtrend AND volume > 2.0 * volume_ma(20)
- Uses Donchian channels from 6h chart for structure-based breakouts
- Weekly EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 12-37 trades/year on 6h) to minimize fee drag
- Exit on opposite Donchian level touch or trend reversal
- Novelty: Uses Donchian breakouts with weekly trend and volume confirmation - different from Camarilla saturation
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
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter (needs completed weekly candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate Donchian channels on 6h chart (primary timeframe)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 20 for Donchian and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
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
            # Long: Price breaks above Donchian high AND weekly uptrend AND volume spike
            if close[i] > donchian_high[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND weekly downtrend AND volume spike
            elif close[i] < donchian_low[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR weekly trend turns down
            if close[i] < donchian_low[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR weekly trend turns up
            if close[i] > donchian_high[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0