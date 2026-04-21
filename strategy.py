#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal breakout with 12h volume confirmation and ADX trend filter.
Longs when price breaks above latest bearish fractal with ADX>20 and volume>1.3x average;
shorts when price breaks below latest bullish fractal with ADX>20 and volume>1.3x average.
Exit on price crossing back through the opposite fractal or 2x ATR stop.
Williams Fractals identify potential reversal points; combining with trend and volume filters
should yield high-probability breakouts in both bull and bear markets while limiting trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Williams Fractals and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Fractals (requires 5 points: t-2, t-1, t, t+1, t+2)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_12h, low_12h)
    # Bearish fractal: high[t] is highest among [t-2, t-1, t, t+1, t+2]
    # Bullish fractal: low[t] is lowest among [t-2, t-1, t, t+1, t+2]
    # These arrays contain the fractal values where they occur, NaN elsewhere
    
    # Calculate 14-period ADX for trend filter on 12h
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(high_12h)
    plus_dm[1:] = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm[1:] = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h indicators to 4h timeframe with extra delay for fractals (need 2 bars confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: volume spike > 1.3x 20-period average on 4h
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period) on 4h
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        bear_fract = bearish_fractal_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above bearish fractal with volume and trend
            if (price_high > bear_fract and 
                adx_val > 20 and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: break below bullish fractal with volume and trend
            elif (price_low < bull_fract and 
                  adx_val > 20 and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite fractal cross OR ATR-based stoploss
            exit_signal = False
            
            # Opposite fractal exit
            if position == 1 and price_close < bull_fract:
                exit_signal = True
            elif position == -1 and price_close > bear_fract:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from fractal level)
            if position == 1:
                # For longs, stop below bullish fractal minus 2x ATR
                if price_close < bull_fract - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above bearish fractal plus 2x ATR
                if price_close > bear_fract + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_12hADX20_Volume1.3x_ATR2x"
timeframe = "4h"
leverage = 1.0