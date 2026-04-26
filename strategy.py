#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike
Hypothesis: Williams fractal breakouts on 6h with 1d EMA50 trend filter and volume confirmation. 
Fractals require 2-bar confirmation (aligned with extra delay) to avoid look-ahead. 
Targets 50-150 total trades over 4 years by requiring confluence of trend, volume, and fractal breakout. 
Works in bull/bear markets via 1d trend filter (EMA50). 
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
    
    # Load 1d data ONCE before loop for HTF trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Williams fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Reduced fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 14 for ATR, 20 for volume median
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Bullish fractal breakout: high > bullish_fractal and volume spike, in uptrend (close > EMA50_1d)
            long_entry = (high[i] > bullish_fractal_aligned[i]) and vol_spike and (close_val > ema_50_val)
            # Bearish fractal breakout: low < bearish_fractal and volume spike, in downtrend (close < EMA50_1d)
            short_entry = (low[i] < bearish_fractal_aligned[i]) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, ATR stoploss, or at bearish fractal (take profit)
            stop_price = entry_price - 2.0 * atr_val
            if close_val < ema_50_val or close_val < stop_price or low[i] < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, ATR stoploss, or at bullish fractal (take profit)
            stop_price = entry_price + 2.0 * atr_val
            if close_val > ema_50_val or close_val > stop_price or high[i] > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0