#!/usr/bin/env python3
"""
1d_WilliamsFractal_Rebound_1wTrend_Volume
Hypothesis: Williams Fractal reversals (bearish fractal for long, bullish for short) on 1d timeframe,
confirmed by 1w trend (EMA34) and volume spike. Works in both bull (buy bearish fractal reversals)
and bear (sell bullish fractal reversals) by trading mean reversion within the trend. Target: 30-100
total trades over 4 years (7-25/year). Uses Williams fractal confirmation requiring 2 extra bars.
"""

name = "1d_WilliamsFractal_Rebound_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1w volume SMA20 for volume confirmation
    volume_1w = df_1w['volume'].values
    vol_sma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    # Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Bearish fractal needs 2 extra 1d bars for confirmation (after the center bar)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume SMA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]) or \
           np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1w volume (scaled to 1d)
        # Approximate 1d volume from 1w: 1w volume / 5 (since 7d/1d = 7, but using 5 for sensitivity)
        vol_1d_approx = vol_sma20_1w_aligned[i] / 5.0
        volume_confirm = volume[i] > 1.5 * vol_1d_approx
        
        if position == 0:
            # Long at bearish fractal (potential bottom) with uptrend and volume
            if bearish_fractal_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short at bullish fractal (potential top) with downtrend and volume
            elif bullish_fractal_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below trend or opposite fractal appears
            if close[i] < ema34_1w_aligned[i] or bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above trend or opposite fractal appears
            if close[i] > ema34_1w_aligned[i] or bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals