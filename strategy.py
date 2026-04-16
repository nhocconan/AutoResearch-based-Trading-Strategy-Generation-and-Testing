#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 1d volume spike and ATR filter.
# Long when price breaks above R1 AND volume > 2x 20-period average AND ATR(14) > 0.5*ATR(50) (expanding volatility).
# Short when price breaks below S1 AND volume > 2x 20-period average AND ATR(14) > 0.5*ATR(50).
# Exit when price crosses the 6h midpoint (R1+S1)/2 OR ATR contraction (ATR14 < 0.3*ATR50).
# Uses discrete position size 0.25. Designed to capture breakouts in both bull and bear markets.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (from previous bar) ===
    # Camarilla levels based on previous bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    midpoint = (R1 + S1) / 2  # Exit level
    
    # === 6h Indicators: Volume Spike (volume > 2x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === 6h Indicators: ATR Filter (expanding volatility) ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean()
    atr_ratio = atr_14 / atr_50
    atr_expanding = atr_ratio > 0.5  # Volatility expanding
    atr_contracting = atr_ratio < 0.3  # Volatility contracting (exit condition)
    
    # Get 1d data once before loop (not used in this version but kept for potential HTF extension)
    # df_1d = get_htf_data(prices, '1d')
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(midpoint[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_exp = atr_expanding[i]
        atr_contr = atr_contracting[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint OR volatility contracts
            if price < midpoint[i] or atr_contr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint OR volatility contracts
            if price > midpoint[i] or atr_contr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND volume spike AND expanding volatility
            if price > R1[i] and vol_spike and atr_exp:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 AND volume spike AND expanding volatility
            elif price < S1[i] and vol_spike and atr_exp:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0