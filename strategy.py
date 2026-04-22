#!/usr/bin/env python3

"""
Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation.
Go long when price breaks above a bearish fractal resistance during weekly uptrend with volume spike.
Go short when price breaks below a bullish fractal support during weekly downtrend with volume spike.
Fractals provide natural support/resistance levels; weekly trend filters for direction; volume confirms breakout strength.
Designed for low trade frequency (12-37/year) by requiring fractal formation, trend alignment, and volume spike.
Works in both bull and bear markets by following weekly trend direction.
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
    
    # Load daily data for fractals - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_daily['high'].values,
        df_daily['low'].values,
    )
    # Needs 2 extra daily bars for confirmation after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_daily, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_daily, bullish_fractal, additional_delay_bars=2
    )
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    weekly_close = df_weekly['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_ma_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_50[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal resistance + weekly uptrend + volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                ema50_weekly_aligned[i] > ema50_weekly_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal support + weekly downtrend + volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  ema50_weekly_aligned[i] < ema50_weekly_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or volume drops
            exit_signal = False
            
            if position == 1:
                # Exit long: weekly downtrend or volume drops below average
                if (ema50_weekly_aligned[i] < ema50_weekly_aligned[i-1] or 
                    volume[i] < vol_ma_50[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: weekly uptrend or volume drops below average
                if (ema50_weekly_aligned[i] > ema50_weekly_aligned[i-1] or 
                    volume[i] < vol_ma_50[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0