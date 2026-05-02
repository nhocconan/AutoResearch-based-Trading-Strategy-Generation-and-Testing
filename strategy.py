#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend filter and 6h Williams Fractals for breakout detection
# Entry: Long when bullish fractal breaks above price AND price > 1d EMA34 (uptrend) AND volume spike
#        Short when bearish fractal breaks below price AND price < 1d EMA34 (downtrend) AND volume spike
# Exit: Close crosses 1d EMA34 (trend reversal) OR opposite fractal forms (momentum shift)
# Williams Fractals require 2-bar confirmation delay (additional_delay_bars=2) for validity
# Works in both bull and bear markets by trading with 1d trend using fractal breakouts
# Target: 75-150 total trades over 4 years (19-38/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_WilliamsFractal_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Williams Fractals (need 2-bar confirmation delay)
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Williams fractals need 2 extra 6h bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish fractal forms (price breaks above recent high) AND price > 1d EMA34 (uptrend) AND volume spike
            if (bullish_fractal_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish fractal forms (price breaks below recent low) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (bearish_fractal_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA34 (trend change) OR bearish fractal forms (momentum loss)
            if close[i] < ema_34_1d_aligned[i] or bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA34 (trend change) OR bullish fractal forms (momentum loss)
            if close[i] > ema_34_1d_aligned[i] or bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals