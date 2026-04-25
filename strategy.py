#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + Weekly EMA34 Trend + Volume Spike
Hypothesis: Williams fractals on weekly timeframe identify major swing points with confirmation delay.
Breakouts above bearish fractal (weekly resistance) or below bullish fractal (weekly support)
with weekly EMA34 trend alignment and volume spike capture institutional moves in 12h timeframe.
Weekly HTF provides strong trend filter suitable for 12h entries, reducing whipsaw in bear markets.
Targets 12-37 trades/year (50-150 total over 4 years) with discrete position sizing.
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
    
    # ATR for volatility filtering (optional)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Weekly data for Williams fractals and EMA34 (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Williams fractals: needs 2 extra bars for confirmation
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Align with additional_delay_bars=2 for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Weekly EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly data (fractals + EMA) and volume MA
    start_idx = max(34, 20) + 20  # extra for fractal confirmation delay and weekly bar alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > bearish_fractal_aligned[i]
        breakout_short = curr_close < bullish_fractal_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: fractal breakout + volume spike + weekly EMA34 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_1w_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on bullish fractal retrace or trend change
            if curr_close < bullish_fractal_aligned[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on bearish fractal retrace or trend change
            if curr_close > bearish_fractal_aligned[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0