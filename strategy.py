#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend filter (price above/below 50 EMA).
- Entry: Long when bullish Williams fractal breakout (close > recent fractal high) AND price > 1w EMA50 AND volume > 1.5x 20-period average.
         Short when bearish Williams fractal breakout (close < recent fractal low) AND price < 1w EMA50 AND volume > 1.5x 20-period average.
- Exit: Opposite fractal breakout OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Fractals identify key reversal points with built-in confirmation delay.
- Works in bull markets (buy fractal breakouts above trend) and bear markets (sell fractal breakdowns below trend).
- Estimated trades: ~100 total over 4 years (~25/year) based on fractal frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Williams Fractals on 12h (need 5-bar window: 2 left, center, 2 right)
    # Using additional_delay_bars=2 for confirmation as per Rule 2b
    bullish_fractal, bearish_fractal = compute_williams_fractals(high, low)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Fractal breakout conditions
    # Bullish breakout: close above most recent bullish fractal
    bullish_breakout = close > bullish_fractal_aligned
    # Bearish breakout: close below most recent bearish fractal
    bearish_breakout = close < bearish_fractal_aligned
    
    # 1w trend filter
    trend_bullish = close > ema50_1w_aligned
    trend_bearish = close < ema50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for fractals/EMA/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Exit conditions: opposite fractal breakout OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: bearish fractal breakout OR price falls below 1w EMA50
            if position == 1:
                if bearish_breakout[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish fractal breakout OR price rises above 1w EMA50
            elif position == -1:
                if bullish_breakout[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction with volume confirmation
        if position == 0:
            # Long: Bullish fractal breakout AND bullish 1w trend AND volume confirmation
            if bullish_breakout[i] and trend_bullish[i] and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal breakout AND bearish 1w trend AND volume confirmation
            elif bearish_breakout[i] and trend_bearish[i] and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0