#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_1dTrend
Hypothesis: 4h Donchian(20) breakout with volume spike and 1d EMA34 trend filter.
- Long when price breaks above 4h Donchian upper (20-period high) AND volume spike AND 1d EMA34 uptrend
- Short when price breaks below 4h Donchian lower (20-period low) AND volume spike AND 1d EMA34 downtrend
- Uses completed 4h bar for breakout (no look-ahead)
- Volume spike confirms institutional participation (2.0x 20-period average on 4h)
- 1d EMA34 filter ensures trading with higher timeframe trend (avoids counter-trend whipsaws)
- Designed for moderate frequency (target 20-50 trades/year) to minimize fee drag and improve test generalization
- Exit on opposite Donchian level touch or trend reversal
- Novelty: Donchian structure + volume confirmation + 1d HTF trend on 4h timeframe (proven pattern from DB)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channels (20-period) from completed 4h bar
    donch_high_20 = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (no additional delay needed for structure)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter (needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian/volume MA, 34 for 1d EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trend_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and 1d trend filter
        if position == 0:
            # Long: Price breaks above Donchian upper AND volume spike AND 1d uptrend
            if close[i] > donch_high_aligned[i] and volume_spike[i] and trend_1d[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND volume spike AND 1d downtrend
            elif close[i] < donch_low_aligned[i] and volume_spike[i] and trend_1d[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian lower OR 1d trend turns down
            if close[i] < donch_low_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian upper OR 1d trend turns up
            if close[i] > donch_high_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_1dTrend"
timeframe = "4h"
leverage = 1.0