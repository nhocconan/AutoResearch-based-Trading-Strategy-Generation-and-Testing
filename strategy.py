#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA(close,50), Bear Power = Low - EMA(close,50).
# Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum building) AND 6h volume > 1.5x 20-period average AND EMA50 rising.
# Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum building) AND 6h volume > 1.5x 20-period average AND EMA50 falling.
# Exit when Elder Ray power crosses zero (Bull Power <= 0 for longs, Bear Power >= 0 for shorts).
# Uses discrete position size 0.25. Elder Ray measures price relative to trend, volume confirms participation, EMA50 slope ensures trend alignment.
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA50 and Elder Ray ===
    # EMA50
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Elder Ray: Bull Power = High - EMA50, Bear Power = Low - EMA50
    bull_power = high - ema50
    bear_power = low - ema50
    
    # EMA50 slope (rising/falling)
    ema50_slope = ema50 - np.roll(ema50, 1)
    ema50_slope[0] = 0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_slope[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema50_val = ema50[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema50_slope_val = ema50_slope[i]
        vol_ma_val = vol_ma_20[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # EMA50 trend filter
        ema_rising = ema50_slope_val > 0
        ema_falling = ema50_slope_val < 0
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power crosses zero (loses bullish momentum)
            if bull_val <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power crosses zero (loses bearish momentum)
            if bear_val >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 (above EMA50) AND Bear Power decreasing (bullish building) AND volume filter AND EMA50 rising
            if bull_val > 0 and i > warmup and bear_val < bear_power[i-1] and vol_filter and ema_rising:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0 (below EMA50) AND Bull Power decreasing (bearish building) AND volume filter AND EMA50 falling
            elif bear_val < 0 and i > warmup and bull_val < bull_power[i-1] and vol_filter and ema_falling:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_EMA50_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0