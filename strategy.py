#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + Weekly EMA34 Trend + Volume Spike
Hypothesis: Williams fractals on weekly timeframe identify significant swing points.
Breakouts above weekly bearish fractal (resistance) or below bullish fractal (support)
with volume confirmation (>2x 20-period volume MA) and alignment with weekly EMA34 trend
capture strong momentum moves. Works in both bull and bear markets via weekly trend filter.
Designed for 6h timeframe targeting 50-150 total trades over 4 years.
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
    
    # Get weekly data for fractals and EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 weeks for EMA34
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Williams fractals (requires 2 extra bars for confirmation)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Align with additional_delay_bars=2 for fractal confirmation (needs 2 future weekly bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for weekly EMA34, volume MA, and aligned fractals
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        # Trend filter: price relative to weekly EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at weekly fractal levels
            # Long: price breaks above weekly bearish fractal (resistance) with volume confirmation in uptrend
            long_breakout = (curr_close > bearish_fractal_val) and volume_confirm and uptrend
            # Short: price breaks below weekly bullish fractal (support) with volume confirmation in downtrend
            short_breakout = (curr_close < bullish_fractal_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit conditions: price closes below weekly bullish fractal OR weekly EMA34 trend turns down
            if curr_close < bullish_fractal_val or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit conditions: price closes above weekly bearish fractal OR weekly EMA34 trend turns up
            if curr_close > bearish_fractal_val or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0