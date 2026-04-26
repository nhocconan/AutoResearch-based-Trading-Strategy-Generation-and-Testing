#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike
Hypothesis: Use 1d Williams fractals to identify swing points and breakout direction,
confirmed by 1d EMA trend and volume spike on 6h. Enter on 6h breakout above/below
recent fractal levels with volume confirmation. Designed for 6h timeframe to work
in both bull and bear markets by following higher timeframe trend with momentum
confirmation. Targets 12-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Williams fractals (needs 2-bar confirmation delay)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: high[2] is highest of high[0:5]
    # Bullish fractal: low[2] is lowest of low[0:5]
    # Align with 2-bar delay for confirmation (fractal confirmed after 2nd following bar closes)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 6h volume spike (volume > 1.5 * 20-period MA)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate 6h ATR for stop loss reference (not used in signal, but for exit logic)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for EMA, enough for fractals (5 bars), 20 for volume MA
    start_idx = max(34, 5, 20) + 2  # +2 for fractal alignment delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for breakout in direction of 1d trend
            # Uptrend: price above 1d EMA34
            # Downtrend: price below 1d EMA34
            uptrend = close_val > ema_34_1d_aligned[i]
            downtrend = close_val < ema_34_1d_aligned[i]
            
            # Bullish breakout: price breaks above nearest bullish fractal resistance
            # Bearish breakout: price breaks below nearest bearish fractal support
            bullish_break = (bullish_fractal_aligned[i] > 0) and (close_val > bullish_fractal_aligned[i])
            bearish_break = (bearish_fractal_aligned[i] > 0) and (close_val < bearish_fractal_aligned[i])
            
            # Enter long in uptrend on bullish break with volume spike
            if uptrend and bullish_break and volume_spike[i]:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Enter short in downtrend on bearish break with volume spike
            elif downtrend and bearish_break and volume_spike[i]:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # Long position - exit on trend reversal or mean reversion
            # Exit if price crosses below 1d EMA (trend change)
            # Or if price drops 2*ATR from entry (profit stop/mean reversion)
            if close_val < ema_34_1d_aligned[i] or close_val < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
                
        elif position == -1:
            # Short position - exit on trend reversal or mean reversion
            # Exit if price crosses above 1d EMA (trend change)
            # Or if price rises 2*ATR from entry (profit stop/mean reversion)
            if close_val > ema_34_1d_aligned[i] or close_val > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0