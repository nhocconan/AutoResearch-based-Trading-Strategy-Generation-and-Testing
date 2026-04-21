#!/usr/bin/env python3
"""
1d_Williams_Fractal_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: Daily Williams fractal breakouts filtered by weekly EMA trend and volume spikes.
In trending markets (price > weekly EMA50): breakout continuation (long above bullish fractal, short below bearish fractal).
In ranging markets: no entries to avoid whipsaw.
Volume confirmation (volume > 1.5x 20-day average) ensures institutional participation.
ATR-based stoploss (2.5x) manages risk in volatile crypto markets.
Designed to work in both bull and bear markets by only trading with the weekly trend.
Timeframe: 1d, uses 1w HTF for trend filter.
Target: 30-100 total trades over 4 years = 7-25/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === Weekly EMA50 for trend filter ===
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily Williams Fractals (requires 5-bar window) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Compute Williams fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Align with 2-bar extra delay for confirmation (fractal needs 2 right-side bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # === Volume spike confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # === ATR (20-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) 
            or np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = vol_spike[i]
        ema_trend = ema_50_1w_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        
        if position == 0:
            # Only trade in trending regime (price above/below weekly EMA50)
            if vol_ok:  # Volume confirmation required
                # Long breakout above bullish fractal in uptrend
                if (price > bull_fractal) and (price > ema_trend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short breakout below bearish fractal in downtrend
                elif (price < bear_fractal) and (price < ema_trend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Fractal reversal exit (optional early exit)
            elif price < bull_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Fractal reversal exit (optional early exit)
            elif price > bear_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Fractal_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0