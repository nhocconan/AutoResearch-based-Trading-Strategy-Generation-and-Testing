#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Ehlers Fisher Transform with 1d volume regime filter and 1w trend filter.
# Fisher Transform identifies turning points in price with Gaussian distribution.
# Long when Fisher crosses above -1.5 (oversold bounce) in uptrend (price > 1w EMA50).
# Short when Fisher crosses below +1.5 (overbought rejection) in downtrend (price < 1w EMA50).
# Volume regime filter: only trade when 1d volume > 1.5x 20-day average to avoid low-volatility chop.
# Designed for low trade frequency (~20-40/year) to minimize fee decay while capturing reversals in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for volume regime (once before loop)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period average volume for regime filter
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 50-period EMA on 1w close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d and 1w indicators to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ehlers Fisher Transform on 4h close (length=9)
    price = prices['close'].values
    hl2 = (prices['high'].values + prices['low'].values) / 2
    max_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).max().values
    min_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).min().values
    range_hl2 = max_hl2 - min_hl2
    value1 = np.where(range_hl2 != 0, 2 * ((hl2 - min_hl2) / range_hl2 - 0.5), 0)
    # Smooth value1 with EMA(4)
    value1_smoothed = pd.Series(value1).ewm(span=4, adjust=False, min_periods=4).mean().values
    # Clamp to [-0.999, 0.999] to avoid domain errors in log
    value1_clamped = np.clip(value1_smoothed, -0.999, 0.999)
    # Fisher Transform: 0.5 * ln((1+value)/(1-value))
    fish = 0.5 * np.log((1 + value1_clamped) / (1 - value1_clamped))
    # Smooth Fisher with EMA(5)
    fish_smoothed = pd.Series(fish).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(fish_smoothed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = price[i]
        vol_ma = vol_ma_aligned[i]
        vol_current = vol_1d[i // 16] if i >= 16 else 0  # Current day's volume (approximation for 4h)
        ema_val = ema_50_aligned[i]
        fish_val = fish_smoothed[i]
        fish_prev = fish_smoothed[i-1]
        
        # Volume regime: current day's volume > 1.5 * 20-day average
        vol_regime = vol_current > 1.5 * vol_ma
        
        # Fisher signals
        fish_cross_up = fish_prev <= -1.5 and fish_val > -1.5
        fish_cross_down = fish_prev >= 1.5 and fish_val < 1.5
        
        if position == 0:
            # Long: Fisher crosses above -1.5 + uptrend + volume regime
            if fish_cross_up and price_close > ema_val and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 + downtrend + volume regime
            elif fish_cross_down and price_close < ema_val and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Fisher crosses below +1.5 or trend breaks
                if fish_cross_down or price_close < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Fisher crosses above -1.5 or trend breaks
                if fish_cross_up or price_close > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_FisherTransform_1dVolRegime_1wEMA50"
timeframe = "4h"
leverage = 1.0