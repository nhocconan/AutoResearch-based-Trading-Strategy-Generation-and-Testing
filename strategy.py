#!/usr/bin/env python3
"""
1d_WilliamsFractal_Alligator_1wTrend_v1
Hypothesis: Daily Williams Fractal signals filtered by weekly Alligator trend (Jaw/Teeth/Lips alignment) and volume confirmation. Enters on bullish/bearish fractal breaks when Alligator is aligned (trending) and volume > 1.5x average. Uses ATR-based stoploss. Designed for low frequency (15-30 trades/year) to minimize fee drag in ranging/bear markets. Works in bull via trend-following breaks and in bear via fade at extremes with Alligator misalignment filtering. Focus on BTC/ETH as primary targets.
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
    
    # Get weekly data for Alligator trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Alligator (Williams) on weekly: Jaw(13,8), Teeth(8,5), Lips(5,3)
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).ewm(span=13, adjust=False).mean().shift(8).values
    teeth = pd.Series(close_1w).ewm(span=8, adjust=False).mean().shift(5).values
    lips = pd.Series(close_1w).ewm(span=5, adjust=False).mean().shift(3).values
    
    # Align Alligator to daily (with extra delay for smoothed indicators)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips, additional_delay_bars=0)
    
    # Get daily data for Williams Fractals and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate volume spike filter: volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate Williams Fractals on daily (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align fractals with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Alligator setup, ATR, volume MA
    start_idx = max(50, 14, 50) + 2
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        # Alligator trend: Jaw > Teeth > Lips = uptrend, reverse = downtrend
        alligator_up = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        alligator_down = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: bullish fractal break AND Alligator uptrend AND volume spike
            long_signal = bullish_fractal_aligned[i] and alligator_up and vol_spike
            
            # Short: bearish fractal break AND Alligator downtrend AND volume spike
            short_signal = bearish_fractal_aligned[i] and alligator_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Alligator flips down OR price hits ATR stoploss
            if (not alligator_up) or (close_val < entry_price - 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Alligator flips up OR price hits ATR stoploss
            if (not alligator_down) or (close_val > entry_price + 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsFractal_Alligator_1wTrend_v1"
timeframe = "1d"
leverage = 1.0