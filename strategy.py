#!/usr/bin/env python3
"""
6h_WilliamsFractal_1dTrend_VolumeReversal_v1
Hypothesis: Trade reversals at 1d Williams Fractals (confirmed with 2-bar delay) in the direction of 1d EMA50 trend, with volume spike confirmation on 6h timeframe. Uses ATR-based trailing stop (2.0x) and requires price >1.5% from EMA50 to avoid chop. Position size 0.25. Designed to capture mean-reversion moves within the dominant daily trend, working in both bull and bear markets by aligning with HTF momentum while exploiting short-term exhaustion at key support/resistance levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractal calculation
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Fractals (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Align with 2-bar additional delay for confirmation (fractal needs 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x median volume (6h timeframe)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR for stop (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Price distance from EMA50 to avoid chop (>1.5%)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_distance = np.abs((close - ema_50_1d_aligned) / ema_50_1d_aligned * 100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (50), volume median (30), 6h ATR (14)
    start_idx = max(50, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(ema_distance[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_val = atr_14[i]
        ema_distance_val = ema_distance[i]
        
        if position == 0:
            # Long: bullish fractal (support), uptrend (close > EMA50), volume spike, price >1.5% from EMA
            long_signal = bullish_fractal_val and \
                          (close_val > ema_50_1d_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (ema_distance_val > 1.5)
            # Short: bearish fractal (resistance), downtrend (close < EMA50), volume spike, price >1.5% from EMA
            short_signal = bearish_fractal_val and \
                           (close_val < ema_50_1d_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (ema_distance_val > 1.5)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50) after minimum holding period
            if bars_since_entry >= 3 and ((low_val < long_stop) or (close_val < ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50) after minimum holding period
            if bars_since_entry >= 3 and ((high_val > short_stop) or (close_val > ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_1dTrend_VolumeReversal_v1"
timeframe = "6h"
leverage = 1.0