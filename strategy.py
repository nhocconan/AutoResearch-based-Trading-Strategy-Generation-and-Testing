#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal breakout with 1d trend filter and volume confirmation.
Long when price breaks above bearish fractal resistance with bullish 1d trend and volume spike.
Short when price breaks below bullish fractal support with bearish 1d trend and volume spike.
Exit when price returns to the fractal level or trend weakens.
Williams Fractals require 2-bar confirmation, so we use additional_delay_bars=2.
Designed for low trade frequency (20-40/year) to minimize fee flood.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Williams Fractals - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_daily['high'].values,
        df_daily['low'].values,
    )
    
    # Fractals need 2 extra bars for confirmation (formed bar + 2 confirmation bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_daily, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_daily, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA34 for trend filter
    close_d = pd.Series(df_daily['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above bearish fractal resistance with bullish 1d trend and volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal support with bearish 1d trend and volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to bullish fractal support OR trend turns bearish
                if close[i] <= bullish_fractal_aligned[i] or close[i] < ema34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to bearish fractal resistance OR trend turns bullish
                if close[i] >= bearish_fractal_aligned[i] or close[i] > ema34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsFractal_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
#%%