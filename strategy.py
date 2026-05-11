#!/usr/bin/env python3
# 6H_Williams_Fractal_Volume_Trend_v1
# Hypothesis: Williams fractals on 1d confirm swing points; 6s trend continuation after fractal breakout
# with volume confirmation provides edge in both bull and bear markets. Targets 50-150 trades over 4 years.

name = "6H_Williams_Fractal_Volume_Trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Williams fractals (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams fractals on 1d (need 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Add 2-bar delay for confirmation (fractal forms after 2 bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 6s EMA trend filter (21-period)
    ema_fast = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean()
    ema_slow = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean()
    ema_fast = ema_fast.values
    ema_slow = ema_slow.values
    
    # Volume spike confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if fractal data is not available
        if np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                # Hold position with trailing stop based on EMA crossover
                if position == 1 and ema_fast[i] < ema_slow[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and ema_fast[i] > ema_slow[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions: breakout after fractal confirmation
        # Bullish: price breaks above recent bearish fractal resistance
        # Bearish: price breaks below recent bullish fractal support
        bullish_breakout = (
            close[i] > bearish_fractal_aligned[i] and
            ema_fast[i] > ema_slow[i] and
            vol_spike[i]
        )
        bearish_breakout = (
            close[i] < bullish_fractal_aligned[i] and
            ema_fast[i] < ema_slow[i] and
            vol_spike[i]
        )
        
        if position == 0:
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on trend reversal or opposite fractal breakout
            if position == 1:
                if (ema_fast[i] < ema_slow[i]) or bearish_breakout:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (ema_fast[i] > ema_slow[i]) or bullish_breakout:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals