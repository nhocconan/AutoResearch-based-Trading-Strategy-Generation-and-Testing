#!/usr/bin/env python3
"""
12h_Williams_Fractal_Breakout_With_Volume_Confirmation
Hypothesis: Williams fractal breakouts on 12h with volume confirmation and 1d EMA trend filter.
Buy when price breaks above bearish fractal (resistance) with volume spike and uptrend.
Sell when price breaks below bullish fractal (support) with volume spike and downtrend.
Designed for low trade frequency (12-37/year) to avoid fee drag while capturing
breakout moves in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Williams fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal with volume spike and uptrend
            if not np.isnan(bear_fractal) and price > bear_fractal and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal with volume spike and downtrend
            elif not np.isnan(bull_fractal) and price < bull_fractal and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below bullish fractal OR trend turns down
            if not np.isnan(bull_fractal) and price < bull_fractal:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above bearish fractal OR trend turns up
            if not np.isnan(bear_fractal) and price > bear_fractal:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Fractal_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0