#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_Volume
Hypothesis: Uses weekly Williams fractal breakouts with 1w EMA trend filter and daily volume confirmation to capture strong institutional moves in both bull and bear markets. Williams fractals identify key swing points where price reverses, and breakouts from these levels with trend and volume confirmation capture momentum moves. Targets 10-20 trades/year on 1d to minimize fee drag while maintaining high win rate in trending markets.
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
    
    # Get weekly data for trend and fractal calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams fractals on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    
    # Fractals need 2 extra weekly bars for confirmation (formed after the pattern)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Daily volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, fractals, and volume MA
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1w_aligned[i]
        bearish_fractal_level = bearish_fractal_aligned[i]
        bullish_fractal_level = bullish_fractal_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above bullish fractal (resistance) with uptrend and volume spike
            if close[i] > bullish_fractal_level and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below bearish fractal (support) with downtrend and volume spike
            elif close[i] < bearish_fractal_level and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: break below bearish fractal or trend turns down
            if close[i] < bearish_fractal_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above bullish fractal or trend turns up
            if close[i] > bullish_fractal_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0