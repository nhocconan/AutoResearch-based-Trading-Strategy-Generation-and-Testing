#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirmation
Hypothesis: 6h breakout above/below daily Williams fractal levels in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA). Williams fractals identify key swing points with inherent confirmation delay (needs 2 extra daily bars). Trend filter ensures alignment with higher timeframe momentum. Volume confirmation reduces false breakouts. Designed for 12-30 trades/year (50-120 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 1d trend while using fractal structure for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams fractals from 1d data (needs 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with additional_delay_bars=2 for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 34 for ema, 2 for fractal alignment)
    start_idx = max(20, 34, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: breakout of fractal level in trend direction with volume
        long_entry = (close_val > bullish_fractal_val) and bullish_1d and vol_spike
        short_entry = (close_val < bearish_fractal_val) and bearish_1d and vol_spike
        
        # Exit conditions: opposite fractal level touch (or trend reversal)
        exit_long = (close_val < bearish_fractal_val) or not bullish_1d
        exit_short = (close_val > bullish_fractal_val) or not bearish_1d
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0