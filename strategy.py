#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d Williams Fractal regime filter and volume confirmation.
- Long: price breaks above Camarilla R3 (1d) + bullish 1d fractal present (uptrend regime) + volume > 1.5x 20-period avg volume
- Short: price breaks below Camarilla S3 (1d) + bearish 1d fractal present (downtrend regime) + volume > 1.5x 20-period avg volume
- Exit: trailing stop (2.0x ATR from extreme) OR Camarilla breakout in opposite direction
- Williams Fractal on 1d identifies regime: bullish fractal = higher low structure, bearish = lower high structure
- Uses Camarilla R3/S3 (wider bands) for fewer, more significant breakouts vs R1/S1
- Volume confirmation (1.5x spike) reduces false breakouts
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: fractal regime adapts to structure
- Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla levels and Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R3, S3) on 1d data
    # Camarilla: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = df_1d['close'] + (1.1 * (df_1d['high'] - df_1d['low']) / 4)
    camarilla_s3 = df_1d['close'] - (1.1 * (df_1d['high'] - df_1d['low']) / 4)
    camarilla_r3_vals = camarilla_r3.values
    camarilla_s3_vals = camarilla_s3.values
    
    # Calculate Williams Fractals on 1d data for regime filter
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractal needs 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Align HTF indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_vals)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Break above Camarilla R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # Break below Camarilla S3
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filters from Williams Fractals
        bullish_regime = bullish_fractal_aligned[i]  # Bullish fractal present = uptrend structure
        bearish_regime = bearish_fractal_aligned[i]  # Bearish fractal present = downtrend structure
        
        if position == 0:
            # Long: Camarilla breakout up + bullish regime + volume spike
            if breakout_up and bullish_regime and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Camarilla breakout down + bearish regime + volume spike
            elif breakout_down and bearish_regime and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Camarilla breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            breakout_down_exit = close[i] < camarilla_s3_aligned[i]
            
            if trailing_stop_long or breakout_down_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from short extreme (trailing stop)
            # 2. Camarilla breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            breakout_up_exit = close[i] > camarilla_r3_aligned[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dWilliamsFractal_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0