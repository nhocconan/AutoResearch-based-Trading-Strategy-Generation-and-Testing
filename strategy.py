#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1w EMA trend filter and volume confirmation.
Long when price breaks above bearish fractal with bullish EMA trend and volume spike.
Short when price breaks below bullish fractal with bearish EMA trend and volume spike.
Exit when price returns to fractal level or EMA trend weakens.
Williams Fractals require 2-bar confirmation, so we use additional_delay_bars=2.
Designed for very low trade frequency (10-25/year) to minimize fee drift.
Works in both bull (breakouts with trend) and bear (fading false breaks).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend
    ema_34_weekly = pd.Series(df_weekly['close'].values).ewm(
        span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Load daily data for Williams Fractals
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Compute Williams Fractals (requires 5-bar window: 2 left, 1 center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_daily['high'].values,
        df_daily['low'].values,
    )
    
    # Align fractals to 12h with 2-bar additional delay for confirmation
    # Fractals need 2 future daily bars to confirm, so +2 delay
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_daily, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_daily, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above bearish fractal with bullish EMA trend and volume
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_aligned[i] and  # Price above EMA = bullish bias
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal with bearish EMA trend and volume
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_aligned[i] and  # Price below EMA = bearish bias
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to fractal level OR price below EMA
                if close[i] <= bearish_fractal_aligned[i] or close[i] < ema_34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to fractal level OR price above EMA
                if close[i] >= bullish_fractal_aligned[i] or close[i] > ema_34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsFractal_EMA34_Volume"
timeframe = "12h"
leverage = 1.0
#%%