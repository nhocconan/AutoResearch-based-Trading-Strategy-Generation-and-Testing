#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams fractals identify key swing points where price reverses. 
Breakouts above bearish fractals (resistance) or below bullish fractals (support) 
with 1d EMA50 trend filter and volume spike capture momentum after swing point failures.
Works in bull (buy fractal breakouts in uptrend) and bear (sell fractal breakdowns in downtrend).
Target 20-30 trades/year on 4h to avoid fee drag.
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
    
    # Get 1d data for fractals and EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with additional_delay_bars=2 for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA50, ATR, volume MA
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Williams fractal levels
            # Long: price breaks above bearish fractal (resistance) with volume confirmation in uptrend
            long_breakout = (curr_close > bear_fractal) and volume_confirm and uptrend
            # Short: price breaks below bullish fractal (support) with volume confirmation in downtrend
            short_breakout = (curr_close < bull_fractal) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below bullish fractal OR 2.0*ATR trailing stop OR EMA50 trend turns down
            if curr_close < bull_fractal or curr_close < (highest_since_entry - 2.0 * atr_val) or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above bearish fractal OR 2.0*ATR trailing stop OR EMA50 trend turns up
            if curr_close > bear_fractal or curr_close > (lowest_since_entry + 2.0 * atr_val) or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0