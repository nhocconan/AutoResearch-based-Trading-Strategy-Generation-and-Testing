#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 1d EMA50 trend filter and volume confirmation.
# Uses daily Williams Fractals to identify swing points, enters on breakout of recent fractal high/low.
# Only trades in direction of 1d EMA50 to avoid counter-trend moves.
# Volume filter ensures breakouts have participation.
# Designed for 6h timeframe with target 15-30 trades/year to minimize fee drag.
# Williams Fractals require 2-bar confirmation (already handled by compute_williams_fractals).
name = "6h_WilliamsFractal_Breakout_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n-2] is highest of [n-4:n-1]
    # Bullish fractal: low[n-2] is lowest of [n-4:n-1]
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d)-2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Need 2-bar confirmation for fractals to be valid (standard definition)
    bearish_fractal_confirmed = np.zeros_like(bearish_fractal, dtype=bool)
    bullish_fractal_confirmed = np.zeros_like(bullish_fractal, dtype=bool)
    bearish_fractal_confirmed[2:] = bearish_fractal[:-2]
    bullish_fractal_confirmed[2:] = bullish_fractal[:-2]
    
    # Align to 6h timeframe with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed.astype(float), additional_delay_bars=2)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish fractal breakout (price > recent bullish fractal low) AND uptrend AND volume
            # Recent bullish fractal low - we use the value at the fractal point
            # For simplicity, we use the low of the bar where fractal occurred
            # Since we don't store the price value, we approximate: bullish fractal acts as support
            # We'll use a simple breakout: price > previous high when bullish fractal confirmed
            # Actually, better: wait for breakout above the high that formed the fractal
            # But we don't have that stored. Alternative: use price > recent high as breakout signal
            # Let's use: bullish fractal confirmed AND price breaking up with volume
            # We need the price level of the fractal. Since we don't have it, we'll use a different approach:
            # Enter long when bullish fractal confirmed AND price > previous bar's high AND uptrend
            # This is not perfect but workable.
            # Instead, let's store the actual price values of fractals.
            # We'll modify: store high/low values at fractal points.
            pass  # We'll implement properly below
            
        # Redesign: instead of boolean arrays, store the actual fractal levels
        
    # Redo fractal detection to store actual price levels
    bearish_fractal_high = np.full(len(high_1d), np.nan)
    bullish_fractal_low = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d)-2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal_high[i] = high_1d[i]  # The high that forms the fractal
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal_low[i] = low_1d[i]   # The low that forms the fractal
    
    # Shift by 2 for confirmation (standard Williams fractal needs 2 bars to confirm)
    bearish_fractal_high_confirmed = np.full(len(bearish_fractal_high), np.nan)
    bullish_fractal_low_confirmed = np.full(len(bullish_fractal_low), np.nan)
    bearish_fractal_high_confirmed[2:] = bearish_fractal_high[:-2]
    bullish_fractal_low_confirmed[2:] = bullish_fractal_low[:-2]
    
    # Align the fractal levels to 6s timeframe
    bearish_fractal_high_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_high_confirmed)
    bullish_fractal_low_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_low_confirmed)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_high_aligned[i]) or np.isnan(bullish_fractal_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above recent bullish fractal low (support) AND uptrend AND volume
            # Actually, bullish fractal low is support, so we wait for breakdown? No.
            # Bullish fractal indicates potential support/resistance change.
            # Standard usage: bullish fractal = potential bottom, so we buy on breakout above it
            # But the fractal low is the low point, so breaking above it means bullish continuation
            # Wait, let's think: bullish fractal forms at a low point, so price should go up from there.
            # We'll enter long when price crosses above the bullish fractal low (which acted as support)
            # AND we're in uptrend (price > EMA50)
            long_cond = (close[i] > bullish_fractal_low_aligned[i]) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            
            # Short conditions: price breaks below recent bearish fractal high (resistance) AND downtrend AND volume
            short_cond = (close[i] < bearish_fractal_high_aligned[i]) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below bearish fractal high (resistance) OR downtrend OR volume filter fails
            if (close[i] < bearish_fractal_high_aligned[i]) or (close[i] < ema50_1d_aligned[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above bullish fractal low (support) OR uptrend OR volume filter fails
            if (close[i] > bullish_fractal_low_aligned[i]) or (close[i] > ema50_1d_aligned[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals