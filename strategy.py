#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Long when Bull Power > 0 AND price > 1d EMA50 (uptrend) AND 6h volume > 1.8x 20-period average.
# Short when Bear Power < 0 AND price < 1d EMA50 (downtrend) AND 6h volume > 1.8x 20-period average.
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength relative to EMA13, 
# 1d EMA50 ensures higher timeframe trend alignment, volume spike confirms institutional participation.
# Designed to work in both bull (buy strength) and bear (sell weakness) markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA13 for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h Indicators: Elder Ray Index ===
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # === 6h Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50, 20 for volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bp = bull_power[i]
        br = bear_power[i]
        price = close[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative (weakening bullishness) or volume spike ends
            if bp <= 0 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns positive (weakening bearishness) or volume spike ends
            if br >= 0 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 (bullish strength) AND price > 1d EMA50 (uptrend) AND volume spike
            if bp > 0 and price > ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 (bearish strength) AND price < 1d EMA50 (downtrend) AND volume spike
            elif br < 0 and price < ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0