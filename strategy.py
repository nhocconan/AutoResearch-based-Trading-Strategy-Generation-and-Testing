#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d regime filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and rising + Bear Power < 0 + volume > 1.3x avg + 1d close > 1d EMA50
# Short when Bear Power < 0 and falling + Bull Power < 0 + volume > 1.3x avg + 1d close < 1d EMA50
# Exit when power signals diverge or volume drops
# Designed for 12-30 trades/year on 6h timeframe with strong trend persistence

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for regime filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Regime filter: price relative to 1d EMA50
        is_uptrend_regime = close[i] > ema_50_1d_aligned[i]
        is_downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray conditions with slope (requiring at least 2 bars for slope)
        if i >= 21:
            bull_rising = bull_power[i] > bull_power[i-1]
            bull_falling = bull_power[i] < bull_power[i-1]
            bear_rising = bear_power[i] > bear_power[i-1]
            bear_falling = bear_power[i] < bear_power[i-1]
        else:
            bull_rising = bull_falling = bear_rising = bear_falling = False
        
        # Entry conditions
        long_entry = (bull_power[i] > 0) and bull_rising and (bear_power[i] < 0) and volume_filter and is_uptrend_regime
        short_entry = (bear_power[i] < 0) and bear_falling and (bull_power[i] < 0) and volume_filter and is_downtrend_regime
        
        # Exit conditions: power divergence or loss of regime/volume
        long_exit = (bull_power[i] <= 0) or (not bull_rising) or (bear_power[i] >= 0) or (not volume_filter) or (not is_uptrend_regime)
        short_exit = (bear_power[i] >= 0) or (not bear_falling) or (bull_power[i] >= 0) or (not volume_filter) or (not is_downtrend_regime)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals