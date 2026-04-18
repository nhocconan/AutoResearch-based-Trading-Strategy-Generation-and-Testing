#!/usr/bin/env python3
"""
12h_Williams_Fractal_Breakout_With_Volume_Confirmation
Hypothesis: Williams Fractal breakouts on 12h chart with volume confirmation and 1d trend filter.
In bull markets, buy bullish fractal breaks; in bear markets, sell bearish fractal breaks.
Uses 1d EMA for trend filter to avoid counter-trend trades. Designed for low trade frequency
(12-37/year) to avoid fee drag while capturing significant momentum moves.
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
    
    # Get 1d data for Williams Fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Add 2 extra bars for confirmation as fractals need 2 future bars to confirm
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(40, 34*2, 20)  # Warmup for fractals, EMA, volume
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: bullish fractal break with volume spike and uptrend (price > EMA34)
            if price > bull_fract and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal break with volume spike and downtrend (price < EMA34)
            elif price < bear_fract and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < bullish fractal level OR trend turns down
            if price < bull_fract or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > bearish fractal level OR trend turns up
            if price > bear_fract or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Fractal_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0