#!/usr/bin/env python3
"""
12h_WilliamsFractal_Breakout_1dTrend_VolumeSpike
Hypothesis: Use daily Williams fractals to identify swing highs/lows. Break above recent bearish fractal (sell signal) or below recent bullish fractal (buy signal) triggers entry, filtered by 1d EMA50 trend and volume spike (2.0x 20-period average). Only trade in direction of 1d trend to avoid whipsaw. ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to limit drawdown. Designed for 12h timeframe to capture multi-day swings in both bull and bear markets. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend and fractals)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Williams Fractals (requires 2-bar confirmation delay) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: sell signal below this level; needs 2 extra bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    # Bullish fractal: buy signal above this level; needs 2 extra bars for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average (strict filter)
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Only enter in direction of 1d trend: long if price > EMA50, short if price < EMA50
            long_condition = (price > bull_fract) and (price > ema_trend) and volume_confirmed
            short_condition = (price < bear_fract) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at bearish fractal (sell area)
            elif price < bear_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at bullish fractal (buy area)
            elif price > bull_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0