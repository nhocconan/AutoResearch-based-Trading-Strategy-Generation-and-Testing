#!/usr/bin/env python3
"""
1d_WilliamsFractal_1wTrend_VolumeBreakout
Hypothesis: Daily timeframe strategy using weekly trend filter (price > weekly EMA34) and Williams Fractal breakouts with volume confirmation (>1.5x 20-bar avg). Enters long on bullish fractal breakout above weekly EMA in uptrend, short on bearish fractal breakout below weekly EMA in downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 1d timeframe with ~10-25 trades/year, works in bull/bear by following weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for HTF trend filter and fractals
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Fractals on weekly data (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Fractals need 2 extra weekly bars for confirmation (total 3-bar delay: 1 for close + 2 for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 34-period data for weekly EMA and fractal calculation
    start_idx = max(34, 10)  # 34 for EMA, plus fractal lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal breakout above weekly EMA in uptrend with volume confirmation
            bullish_setup = bullish_fractal_aligned[i] and (close[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: bearish fractal breakout below weekly EMA in downtrend with volume confirmation
            bearish_setup = bearish_fractal_aligned[i] and (close[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish fractal breakout OR price crosses below weekly EMA
            if bearish_fractal_aligned[i] or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish fractal breakout OR price crosses above weekly EMA
            if bullish_fractal_aligned[i] or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsFractal_1wTrend_VolumeBreakout"
timeframe = "1d"
leverage = 1.0