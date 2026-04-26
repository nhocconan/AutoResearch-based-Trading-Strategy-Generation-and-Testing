#!/usr/bin/env python3
"""
12h_WilliamsFractal_Donchian20_1wTrend_VolumeSpike_v1
Hypothesis: Williams fractal (bearish for shorts, bullish for longs) combined with 1w Donchian(20) trend filter and 1d volume spike captures reversal points in the direction of the weekly trend. Works in bull/bear by only taking fractal signals aligned with weekly Donchian trend. Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Donchian trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w Donchian(20) for trend filter
    donch_high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    
    # Load 1d data ONCE before loop for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Williams fractals: need 5 bars (2 left, 1 center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractal needs 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian(20) (20), volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        bullish_fract = bullish_fractal_aligned[i]
        bearish_fract = bearish_fractal_aligned[i]
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(bullish_fract) or np.isnan(bearish_fract) or 
            np.isnan(donch_high) or np.isnan(donch_low)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price above Donchian high = uptrend, price below Donchian low = downtrend
        is_uptrend = close_val > donch_high
        is_downtrend = close_val < donch_low
        
        # Entry conditions: Williams fractal in direction of trend + volume confirmation
        long_condition = bullish_fract and is_uptrend and vol_conf
        short_condition = bearish_fract and is_downtrend and vol_conf
        
        # Exit conditions: opposite fractal touch or trend reversal
        long_exit = (position == 1 and (bearish_fract or not is_uptrend))
        short_exit = (position == -1 and (bullish_fract or not is_downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_WilliamsFractal_Donchian20_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0