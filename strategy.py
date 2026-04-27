#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm
Hypothesis: Uses daily Williams Fractals for key support/resistance levels on 6h timeframe.
Enter long when price breaks above the most recent bullish fractal AND 1d close > EMA34 (uptrend) AND volume > 1.5 * 20-period average.
Enter short when price breaks below the most recent bearish fractal AND 1d close < EMA34 (downtrend) AND volume > 1.5 * 20-period average.
Exit when price returns to the pivot level (fractal level) OR trend reverses.
Williams Fractals identify significant swing points that act as support/resistance.
Combined with 1d trend filter and volume confirmation, this should work in both bull and bear markets by following higher timeframe structure.
Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size to manage drawdown.
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
    
    # Get 1d data for Williams Fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams Fractals need 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), fractals (5+2)
    start_idx = max(34, 20, 5+2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        bear_level = bearish_fractal_aligned[i]
        bull_level = bullish_fractal_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Williams Fractal levels with 1d trend filter AND volume
            # Long: price breaks above bullish fractal (resistance) AND 1d uptrend AND volume
            long_condition = (close_val > bull_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below bearish fractal (support) AND 1d downtrend AND volume
            short_condition = (close_val < bear_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to bullish fractal level OR trend breaks
            exit_condition = (close_val <= bull_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to bearish fractal level OR trend breaks
            exit_condition = (close_val >= bear_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0