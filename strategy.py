#!/usr/bin/env python3
# Hypothesis: 6-hour Williams Fractal with daily trend filter and volume confirmation.
# Fractals identify potential reversal points at swing highs/lows. In trending markets
# (price above/below daily EMA50), we trade breakouts in the direction of the trend.
# Volume confirms the breakout strength. This combines mean-reversion (fractal rejection)
# with trend-following (breakout continuation) to work in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_WilliamsFractal_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Fractals (need 2-bar confirmation for daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Additional 2-bar delay for fractal confirmation (needs 2 future daily bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish fractal (support hold) + price above daily EMA50 + volume confirmation
            if bullish_fractal_aligned[i] and (close[i] > ema_50_aligned[i]) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal (resistance hold) + price below daily EMA50 + volume confirmation
            elif bearish_fractal_aligned[i] and (close[i] < ema_50_aligned[i]) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below daily EMA50 OR bearish fractal forms (resistance)
            if (close[i] < ema_50_aligned[i]) or bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above daily EMA50 OR bullish fractal forms (support)
            if (close[i] > ema_50_aligned[i]) or bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals