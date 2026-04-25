#!/usr/bin/env python3
"""
1d Williams Fractal Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Fractals identify key swing highs/lows on 1d timeframe. Breakouts above
bearish fractals (short) or below bullish fractals (long) capture momentum with 1w EMA50 trend filter.
Volume confirmation ensures breakout validity. Chop filter (BBW percentile > 0.5) avoids low-volatility false breakouts.
Designed for 30-100 trades over 4 years on 1d timeframe. Works in bull/bear via trend filter and regime avoidance.
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
    
    # Load 1d data ONCE before loop for Williams Fractals, EMA50 trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Williams Fractals (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Bearish fractal: high[i-2] < high[i-1] and high[i] < high[i-1] and high[i+1] < high[i-1] and high[i+2] < high[i-1]
    # Bullish fractal: low[i-2] > low[i-1] and low[i] > low[i-1] and low[i+1] > low[i-1] and low[i+2] > low[i-1]
    # Align with 2 extra delay bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d Bollinger Bands for chop regime (20, 2)
    close_1d = df_1d['close'].values
    bb_ma = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_percentile = bb_width / (bb_width_ma + 1e-10)
    chop_filter = bb_width_percentile > 0.5  # Avoid low volatility squeeze
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 50) + 2  # EMA50, BB, plus fractal delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(chop_filter_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Fractal breakout + trend + volume + chop filter
            # Long: price breaks below bullish fractal (support break) AND bullish bias AND volume spike AND chop filter
            # Note: bullish fractal is a support level, break below indicates potential short, but we want long on break above?
            # Correction: Williams Fractals - bullish fractal is a peak (resistance), bearish fractal is a trough (support)
            # Long: price breaks above bearish fractal (resistance break) AND bullish bias
            # Short: price breaks below bullish fractal (support break) AND bearish bias
            long_entry = (curr_high > bearish_fractal_aligned[i]) and bullish_bias and vol_spike and chop_filter_aligned[i]
            short_entry = (curr_low < bullish_fractal_aligned[i]) and bearish_bias and vol_spike and chop_filter_aligned[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below bullish fractal (support break) OR loss of bullish bias
            if (curr_low < bullish_fractal_aligned[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above bearish fractal (resistance break) OR loss of bearish bias
            if (curr_high > bearish_fractal_aligned[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0