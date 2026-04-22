#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Fractal Breakout with 1-day EMA trend filter and volume confirmation.
Trades breakouts of daily Williams fractal levels (bearish fractal for long, bullish for short) in the direction of the daily EMA trend.
Uses volume spike to confirm institutional interest at key fractal levels. Designed for low trade frequency
(12-37 trades/year) to minimize fee drag and work in both bull and bear markets by aligning with
higher timeframe trend and using mean-reversion at extreme fractal levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for fractal calculation and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Williams fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_ltf_to_htf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above bearish fractal (resistance) with uptrend bias
            if not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) with downtrend bias
            elif not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite fractal level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below bullish fractal or closes below daily EMA
                if not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i]:
                    exit_signal = True
                elif close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above bearish fractal or closes above daily EMA
                if not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i]:
                    exit_signal = True
                elif close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0