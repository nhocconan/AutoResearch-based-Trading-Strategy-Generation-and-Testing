#!/usr/bin/env python3
"""
6h_WilliamsFractal_Donchian20_1dTrend_VolumeSpike_v1
Hypothesis: Combine 1d Williams fractal breakouts with 6h Donchian(20) and 1d EMA50 trend filter + volume confirmation.
Williams fractals provide swing high/low structure; Donchian breakouts catch momentum; 1d EMA50 ensures trend alignment.
Designed for 6h to target 12-37 trades/year with discrete sizing (0.25). Works in bull/bear via 1d trend filter.
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
    
    # Load 1d data ONCE before loop for fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d Williams Fractals (need 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar extra delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian(20) - using current timeframe prices
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian(20), EMA50(50), volume MA(20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_1d_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(bullish_fractal_val) or np.isnan(bearish_fractal_val) or
            np.isnan(dch_high) or np.isnan(dch_low)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: bullish fractal break above Donchian high with 1d uptrend and volume spike
        long_condition = (
            high_val > dch_high and  # break above Donchian high
            bullish_fractal_val > 0 and  # confirmed bullish fractal
            uptrend and
            vol_spike
        )
        # Short: bearish fractal break below Donchian low with 1d downtrend and volume spike
        short_condition = (
            low_val < dch_low and  # break below Donchian low
            bearish_fractal_val > 0 and  # confirmed bearish fractal
            downtrend and
            vol_spike
        )
        
        # Exit: price re-enters Donchian channel
        long_exit = (position == 1 and close_val < dch_high)
        short_exit = (position == -1 and close_val > dch_low)
        
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

name = "6h_WilliamsFractal_Donchian20_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0