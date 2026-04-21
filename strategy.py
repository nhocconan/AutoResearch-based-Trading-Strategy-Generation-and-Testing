#!/usr/bin/env python3
"""
6h_WilliamsFractal_TrendRegime_v1
Hypothesis: Williams fractals on 1d timeframe combined with 1w trend regime (EMA50) provide high-probability breakout entries on 6h timeframe. Uses volume confirmation and ATR-based stops. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Load 1w data once for trend regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams fractals on 1d (requires 2 extra bars for confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Additional delay of 2 bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1w EMA50 for trend regime
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stoploss (6h timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Trend regime: bullish if price > weekly EMA50, bearish if price < weekly EMA50
        bullish_regime = close[i] > ema_50_1w_aligned[i]
        bearish_regime = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: bullish fractal breakout in bullish regime with volume
            if bullish_regime and volume_ok:
                if bullish_fractal_aligned[i] and price > high_1d[-1]:  # Use last known fractal high
                    signals[i] = 0.25
                    position = 1
            # Short: bearish fractal breakout in bearish regime with volume
            elif bearish_regime and volume_ok:
                if bearish_fractal_aligned[i] and price < low_1d[-1]:  # Use last known fractal low
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches bearish fractal level or stoploss
            if bearish_fractal_aligned[i] and price < low_1d[-1]:
                signals[i] = 0.0
                position = 0
            elif price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches bullish fractal level or stoploss
            if bullish_fractal_aligned[i] and price > high_1d[-1]:
                signals[i] = 0.0
                position = 0
            elif price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_TrendRegime_v1"
timeframe = "6h"
leverage = 1.0