#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when bullish fractal breaks above R1 AND 1d EMA50 rising AND 6h volume > 1.8x 20-period MA.
Short when bearish fractal breaks below S1 AND 1d EMA50 falling AND 6h volume > 1.8x 20-period MA.
Exit when price touches opposite fractal level or 1d EMA50 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams Fractals provide swing high/low structure, 1d EMA50 filters major trend, volume spike avoids low-momentum breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate Williams Fractals (5-bar window)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Need 2 extra bars for confirmation (fractal confirmed after 2nd following bar closes)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(4, 50, 20)  # Fractals (4), EMA50 (50), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bull_fract = bullish_fractal_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.8x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: Price above bullish fractal AND EMA50 rising AND volume filter
            if price > bull_fract and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price below bearish fractal AND EMA50 falling AND volume filter
            elif price < bear_fract and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches bearish fractal OR EMA50 starts falling
                if price < bear_fract or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches bullish fractal OR EMA50 starts rising
                if price > bull_fract or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractal_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0