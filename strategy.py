#!/usr/bin/env python3
"""
1d_WilliamsFractal_Retest_Breakout_V1
Hypothesis: Daily Williams fractals identify swing points; price retesting broken fractal levels with volume confirmation provides high-probability continuation. Works in bull/bear by using fractal direction as trend filter. Weekly timeframe filters out noise and confirms higher timeframe trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Apply 2-bar confirmation delay for fractals (need 2 candles after to confirm)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or \
           np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        bear_fract = bearish_fractal_confirmed[i]
        bull_fract = bullish_fractal_confirmed[i]
        weekly_ema = ema_34_1w_aligned[i]
        vol_avg = vol_avg_20_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = vol > 1.5 * vol_avg if vol_avg > 0 else False
        
        if position == 0:
            # Long setup: price breaks above bearish fractal resistance with volume, weekly uptrend
            if not np.isnan(bear_fract) and price > bear_fract and vol_confirm and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short setup: price breaks below bullish fractal support with volume, weekly downtrend
            elif not np.isnan(bull_fract) and price < bull_fract and vol_confirm and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below bullish fractal support or weekly trend turns down
            if not np.isnan(bull_fract) and price < bull_fract or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above bearish fractal resistance or weekly trend turns up
            if not np.isnan(bear_fract) and price > bear_fract or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Retest_Breakout_V1"
timeframe = "1d"
leverage = 1.0