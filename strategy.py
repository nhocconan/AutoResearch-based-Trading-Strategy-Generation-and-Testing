#!/usr/bin/env python3
"""
6h_WilliamsFractal_Donchian_Breakout_V1
Hypothesis: On 6h timeframe, combine Williams Fractals (from 1d) with 6h Donchian(20) breakouts.
Entry: Long when price breaks above 6h Donchian upper channel AND a bullish fractal formed on prior 1d candle.
Short when price breaks below 6h Donchian lower channel AND a bearish fractal formed on prior 1d candle.
Volume confirmation (>1.3x 20-period average) filters weak breakouts.
ATR(14) trailing stop via signal=0 when price moves against position by 2.0*ATR.
Designed for low trade frequency (target: 12-25 trades/year) to minimize fee drag and work in both bull/bear markets
via fractal confirmation of swing points and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Williams Fractals)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 bars for fractals
        return np.zeros(n)
    
    # === 1d Williams Fractals with 2-bar confirmation delay ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Fractals need 2 extra 1d bars for confirmation (Williams fractal definition)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # === 6h Indicators (primary timeframe) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Donchian Channel (20-period) for breakouts
    donchian_period = 20
    upper_channel = pd.Series(high_6h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_6h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) 
            or np.isnan(volume_threshold[i]) or np.isnan(atr[i]) 
            or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + bullish fractal on prior 1d
            if price > upper_channel[i] and volume_6h[i] > volume_threshold[i] and bullish_fractal_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian + volume confirmation + bearish fractal on prior 1d
            elif price < lower_channel[i] and volume_6h[i] > volume_threshold[i] and bearish_fractal_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below upper channel (breakout failed)
            elif price < upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above lower channel (breakout failed)
            elif price > lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Donchian_Breakout_V1"
timeframe = "6h"
leverage = 1.0