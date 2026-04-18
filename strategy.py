#!/usr/bin/env python3
"""
12h Williams Fractal Breakout with Volume Spike and Daily Trend Filter
Hypothesis: Williams Fractals identify significant support/resistance levels.
Breakouts with volume confirmation and daily EMA50 trend filter capture
momentum moves. Designed for 12-37 trades/year on 12h timeframe.
"""

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
    
    # Get daily data for Williams fractals (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_d['high'].values,
        df_d['low'].values,
    )
    
    # Align fractals with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_d, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_50)
    
    # Volume spike: 2x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (12h ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bearish_level = bearish_fractal_aligned[i]
        bullish_level = bullish_fractal_aligned[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above bullish fractal with volume spike and price above EMA50 (uptrend)
            if price > bullish_level and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below bearish fractal with volume spike and price below EMA50 (downtrend)
            elif price < bearish_level and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to bearish fractal or ATR trailing stop
            if price <= bearish_level or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to bullish fractal or ATR trailing stop
            if price >= bullish_level or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WilliamsFractal_Breakout_VolumeSpike_EMA50"
timeframe = "12h"
leverage = 1.0