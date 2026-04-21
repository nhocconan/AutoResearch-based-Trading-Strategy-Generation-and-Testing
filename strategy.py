#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal Breakout with 1d Trend Filter and Volume Spike.
Williams Fractals identify key swing points. Breakout above bearish fractal or below bullish
fractal with volume confirmation and aligned daily trend captures momentum. Uses 1d EMA34 for
trend filter to avoid counter-trend trades. Volume spike (>1.5x average) confirms breakout.
Designed for fewer trades (~20-40/year) to minimize fee drift, works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend, fractals, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Fractals (need 2 extra bars for confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above bearish fractal (resistance), volume spike, uptrend
            if (not np.isnan(bearish_fractal_val) and 
                price_close > bearish_fractal_val and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below bullish fractal (support), volume spike, downtrend
            elif (not np.isnan(bullish_fractal_val) and 
                  price_close < bullish_fractal_val and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to the opposing fractal or trend reversal
            if position == 1 and (not np.isnan(bullish_fractal_val) and price_close < bullish_fractal_val):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (not np.isnan(bearish_fractal_val) and price_close > bearish_fractal_val):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0