#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_12hTrend_VolumeFilter_v1
Hypothesis: Williams Fractal breakouts (above bearish fractal or below bullish fractal) on 6h,
filtered by 12h EMA50 trend and volume spike (>1.8x 24-period average). Uses ATR(14) stoploss (2.0x)
and discrete position sizing (0.25) to minimize fee churn. Williams fractals require 2-bar
confirmation delay on HTF, ensuring no look-ahead. Target: 12-37 trades/year per symbol for low
fee drag and strong test generalization across bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Williams fractals and EMA50 trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === Williams Fractals on 12h (requires 2-bar confirmation delay) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_12h, low_12h)
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    # Need 2 extra bars after center for confirmation
    bearish_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # === 12h EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) 
            or np.isnan(ema_50_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 1.8x 24-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
            vol_filter = volume[i] > 1.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > bearish fractal (breakabove), 12h uptrend, volume filter
            long_breakout = price > bearish_aligned[i]
            long_trend = price > ema_50_aligned[i]
            
            # Short conditions: price < bullish fractal (breakdown), 12h downtrend, volume filter
            short_breakout = price < bullish_aligned[i]
            short_trend = price < ema_50_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below bullish fractal (support)
            elif price < bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above bearish fractal (resistance)
            elif price > bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0