#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above latest bullish fractal AND 1w EMA50 rising AND 6h volume > 1.8x 20-period MA.
Short when price breaks below latest bearish fractal AND 1w EMA50 falling AND 6h volume > 1.8x 20-period MA.
Exit when price touches opposite fractal level or 1w EMA50 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades on 6h, volume spike for momentum confirmation.
Williams Fractals provide natural support/resistance levels that work in ranging and trending markets.
Target: 75-175 total trades over 4 years (19-44/year) for 6h timeframe.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate 6h Williams Fractals (5-bar: 2 left, center, 2 right)
    # Need 4 bars to the right for confirmation, so we shift by 2
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Additional delay of 2 bars for fractal confirmation (needs 2 future candles to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Fractals need 4 bars lookback + 2 bars confirmation delay = 6 bars min
    start_idx = max(6, 50, 20)  # Fractals, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
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
            # Long: Break above bullish fractal AND EMA50 rising AND volume filter
            if not np.isnan(bull_fractal) and price > bull_fractal and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below bearish fractal AND EMA50 falling AND volume filter
            elif not np.isnan(bear_fractal) and price < bear_fractal and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches bearish fractal OR EMA50 starts falling
                if not np.isnan(bear_fractal) and price < bear_fractal:
                    exit_signal = True
                elif i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches bullish fractal OR EMA50 starts rising
                if not np.isnan(bull_fractal) and price > bull_fractal:
                    exit_signal = True
                elif i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0