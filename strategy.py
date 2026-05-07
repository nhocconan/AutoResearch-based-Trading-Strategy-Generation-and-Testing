#!/usr/bin/env python3
name = "12h_WilliamsFractal_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams fractals (need 5 candles: 2 on each side)
    high_series = pd.Series(df_1d['high'].values)
    low_series = pd.Series(df_1d['low'].values)
    
    # Bearish fractal: high[i] is highest in [i-2, i-1, i, i+1, i+2]
    bearish_fractal = np.zeros(len(df_1d), dtype=bool)
    bullish_fractal = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(2, len(df_1d) - 2):
        if (df_1d['high'].values[i] >= df_1d['high'].values[i-2] and
            df_1d['high'].values[i] >= df_1d['high'].values[i-1] and
            df_1d['high'].values[i] >= df_1d['high'].values[i+1] and
            df_1d['high'].values[i] >= df_1d['high'].values[i+2]):
            bearish_fractal[i] = True
        if (df_1d['low'].values[i] <= df_1d['low'].values[i-2] and
            df_1d['low'].values[i] <= df_1d['low'].values[i-1] and
            df_1d['low'].values[i] <= df_1d['low'].values[i+1] and
            df_1d['low'].values[i] <= df_1d['low'].values[i+2]):
            bullish_fractal[i] = True
    
    # Williams fractals need 2 extra bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~2 days for 12h to reduce trades
    
    start_idx = max(200, 20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        trend_up = close > ema_50_1w_aligned[i]
        trend_down = close < ema_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Bullish fractal breakout in uptrend with strong volume
            if (bullish_fractal_aligned[i] > 0.5 and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Bearish fractal breakout in downtrend with strong volume
            elif (bearish_fractal_aligned[i] > 0.5 and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price closes below weekly EMA50 or opposite fractal appears
            if (close[i] < ema_50_1w_aligned[i] or 
                bearish_fractal_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above weekly EMA50 or opposite fractal appears
            if (close[i] > ema_50_1w_aligned[i] or 
                bullish_fractal_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using 12h timeframe with Williams fractal breakouts, weekly EMA50 trend filter,
# and volume confirmation will yield 12-37 trades per year (50-150 total over 4 years).
# Williams fractals identify key reversal points, and trading them in the direction of
# the weekly trend captures institutional breakouts. Volume filter ensures breakouts
# have institutional participation. Position size of 0.25 manages drawdown, and cooldown
# of 4 bars (2 days) prevents overtrading. Williams fractals require 2 extra bars for
# confirmation, handled correctly via additional_delay_bars=2 in align_htf_to_ltf.
# This strategy targets BTC and ETH primarily, with secondary application to SOL.