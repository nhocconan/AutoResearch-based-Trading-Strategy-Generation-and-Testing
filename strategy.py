#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above R1 AND 1d volume > 2.0x 20-period average AND 1d CHOP > 61.8 (range market).
# Short when price breaks below S1 AND 1d volume > 2.0x 20-period average AND 1d CHOP > 61.8 (range market).
# Exit when price crosses Camarilla H/L (close) OR volume drops below average OR CHOP < 38.2 (trending).
# Uses discrete position size 0.25. Designed to capture mean-reversion bounces in range-bound markets.
# Target: 50-150 trades over 4 years (12-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous 12h bar) ===
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # H/L = close (pivot point)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_range = prev_high - prev_low
    R1 = prev_close + 1.1 * pivot_range / 12
    S1 = prev_close - 1.1 * pivot_range / 12
    H_L = prev_close  # Camarilla H/L equals close (pivot)
    
    # === 12h Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get 1d data once before loop for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for CHOP calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(ATR1) / (n * log10(n))) / log10(n)
    # where ATR1 = True Range, n = 14
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # True Range highest high and lowest low over period
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_values = chop.values
    
    # Align 1d CHOP to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for MA, 14 for CHOP)
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(H_L[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below H/L (pivot) OR volume spike ends OR CHOP < 38.2 (trending)
            if price < H_L[i] or not vol_spike or chop_val < 38.2:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above H/L (pivot) OR volume spike ends OR CHOP < 38.2 (trending)
            if price > H_L[i] or not vol_spike or chop_val < 38.2:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND volume spike AND CHOP > 61.8 (range market)
            if price > R1[i] and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 AND volume spike AND CHOP > 61.8 (range market)
            elif price < S1[i] and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0