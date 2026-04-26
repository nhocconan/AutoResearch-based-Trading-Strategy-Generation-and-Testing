#!/usr/bin/env python3
"""
1d_WilliamsFractal_Alligator_1wTrend_v1
Hypothesis: Daily Williams Fractal (bearish for short, bullish for long) with 1-week Alligator (Jaw/Teeth/Lips) trend filter and volume confirmation. Works in bull markets by buying fractal breakdowns in uptrends (Alligator aligned up) and in bear markets by selling fractal breakouts in downtrends (Alligator aligned down). Uses discrete position sizing (0.25) to limit fee drag and target 20-60 trades/year. Williams Fractals provide high-probability reversal points; Alligator confirms trend alignment to avoid counter-trend whipsaws. Volume spike ensures institutional participation. Designed for low turnover to thrive in 2025's bearish/range-bound BTC/ETH market.
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
    
    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get 1w data for Alligator trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d (requires 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams Fractals need 2 extra 1d bars for confirmation (center + 2 right)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate Alligator on 1w: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    close_1w = df_1w['close'].values
    # SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        res = np.full(len(arr), np.nan)
        # First value is SMA
        res[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    jaw = smma(close_1w, 13)
    teeth = smma(close_1w, 8)
    lips = smma(close_1w, 5)
    
    # Align Alligator lines to 1d (need completed 1w bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Alligator trend: Mouth open (Lips > Teeth > Jaw) = uptrend, Mouth open down = downtrend
    alligator_up = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    alligator_down = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # Volume confirmation: volume > 2.0 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of fractal delay (2), Alligator (13), volume MA (50)
    start_idx = max(2, 13, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
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
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: bullish fractal AND Alligator up AND volume spike
            long_signal = bullish_fractal_aligned[i] and alligator_up[i] and vol_spike
            
            # Short: bearish fractal AND Alligator down AND volume spike
            short_signal = bearish_fractal_aligned[i] and alligator_down[i] and vol_spike
            
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
            # Exit: Alligator flips down OR close below Teeth (profit protection)
            if (not alligator_up[i]) or (close_val < teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Alligator flips up OR close above Teeth (profit protection)
            if (not alligator_down[i]) or (close_val > teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsFractal_Alligator_1wTrend_v1"
timeframe = "1d"
leverage = 1.0