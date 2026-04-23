#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation.
Long when bullish fractal breakout above prior fractal high AND 1d EMA50 rising AND volume > 1.5x 20-period MA.
Short when bearish fractal breakout below prior fractal low AND 1d EMA50 falling AND volume > 1.5x 20-period MA.
Exit when opposite fractal breaks or 1d EMA50 reverses.
Williams Fractals identify swing points with built-in confirmation (2-bar delay).
1d EMA50 filters major trend to avoid counter-trend trades.
Volume confirmation ensures breakout has momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Fractals (requires 5-bar window: 2 left, center, 2 right)
    # Need extra delay for confirmation as per Rule 2b
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Align with 2 extra bars delay for fractal confirmation (needs 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, None, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, None, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    last_bullish_fractal = np.nan  # track most recent bullish fractal level
    last_bearish_fractal = np.nan  # track most recent bearish fractal level
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50, volume MA (fractals handled via alignment)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_20[i]
        ema_val = ema_50_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        
        # Update fractal levels when new fractals form (not NaN)
        if not np.isnan(bull_fractal):
            last_bullish_fractal = bull_fractal
        if not np.isnan(bear_fractal):
            last_bearish_fractal = bear_fractal
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Bullish fractal breakout AND EMA50 rising AND volume filter
            # Break above prior bullish fractal level (if exists)
            if (not np.isnan(last_bullish_fractal) and 
                price > last_bullish_fractal and 
                ema_rising and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal breakdown AND EMA50 falling AND volume filter
            # Break below prior bearish fractal level (if exists)
            elif (not np.isnan(last_bearish_fractal) and 
                  price < last_bearish_fractal and 
                  ema_falling and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price breaks below prior bearish fractal OR EMA50 starts falling
                if (not np.isnan(last_bearish_fractal) and price < last_bearish_fractal) or \
                   (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price breaks above prior bullish fractal OR EMA50 starts rising
                if (not np.isnan(last_bullish_fractal) and price > last_bullish_fractal) or \
                   (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsFractal_Breakout_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0