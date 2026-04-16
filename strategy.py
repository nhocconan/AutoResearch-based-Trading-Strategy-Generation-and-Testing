#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
# Long when Bear Power < 0 AND price > 12h EMA34 AND volume > 1.5x 20-period average.
# Short when Bull Power > 0 AND price < 12h EMA34 AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength relative to EMA,
# 12h EMA34 provides higher timeframe trend filter, volume spike confirms participation.
# Designed to work in both bull (buy strength) and bear (sell weakness) markets.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA34 for Elder Ray ===
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h Indicators: Bull Power and Bear Power ===
    bull_power = high - ema34
    bear_power = low - ema34
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA34 for trend filter ===
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMAs, 20 for volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema12h_val = ema34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bear Power becomes positive (bulls taking over) or volume spike ends
            if bear_val >= 0 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bull Power becomes negative (bears taking over) or volume spike ends
            if bull_val <= 0 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bear Power < 0 AND price > 12h EMA34 AND volume spike
            if bear_val < 0 and price > ema12h_val and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bull Power > 0 AND price < 12h EMA34 AND volume spike
            elif bull_val > 0 and price < ema12h_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0