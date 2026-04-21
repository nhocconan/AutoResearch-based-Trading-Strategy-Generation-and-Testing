#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams Fractal reversals with 1w EMA trend filter and volume confirmation.
Bullish fractal (support) triggers longs when price > 1w EMA50; bearish fractal (resistance) triggers shorts when price < 1w EMA50.
Volume must exceed 1.5x 20-period average to confirm reversal strength.
Exit on opposite fractal touch or 2x ATR stop.
Designed for 15-25 trades/year (60-100 total over 4 years) to minimize fee fade while capturing reversal points in ranging markets.
Works in ranging markets via fade of fakeouts and in trending markets via pullbacks to EMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Fractals (5-bar pattern: high[2] > high[1] and high[2] > high[3] and high[2] > high[0] and high[2] > high[4])
    # Bearish fractal: high[2] is highest of 5 bars
    bearish_fractal = np.zeros(len(high_1d))
    bullish_fractal = np.zeros(len(low_1d))
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals require 2-bar confirmation after the pattern
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_val = ema_50_1w_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price touches bullish fractal (support) and above weekly EMA50 with volume
            if (price_low <= bullish_fractal_val and bullish_fractal_val > 0 and 
                price_close > ema_val and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches bearish fractal (resistance) and below weekly EMA50 with volume
            elif (price_high >= bearish_fractal_val and bearish_fractal_val > 0 and 
                  price_close < ema_val and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite fractal touch OR ATR-based stoploss
            exit_signal = False
            
            # Opposite fractal exit
            if position == 1 and price_high >= bearish_fractal_val and bearish_fractal_val > 0:
                exit_signal = True
            elif position == -1 and price_low <= bullish_fractal_val and bullish_fractal_val > 0:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry area - using fractal level as reference)
            if position == 1:
                if bullish_fractal_val > 0 and price_close < bullish_fractal_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                if bearish_fractal_val > 0 and price_close > bearish_fractal_val + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsFractal_Reversal_1wEMA50_Volume1.5x_ATR2x"
timeframe = "4h"
leverage = 1.0