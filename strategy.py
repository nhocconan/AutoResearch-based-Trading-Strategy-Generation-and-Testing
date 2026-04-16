#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1-day Williams %R mean reversion + volume confirmation
# Long when: 12h Choppiness > 61.8 (range) AND 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-period average
# Short when: 12h Choppiness > 61.8 (range) AND 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-period average
# Exit when Choppiness < 38.2 (trending) or opposite Williams %R signal
# Designed for low trade frequency (target: 50-150 total trades over 4 years) on 12h timeframe
# Choppiness filter avoids whipsaws in trends, Williams %R captures mean reversion in ranges, volume adds conviction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Choppiness Index (14-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    denominator = max_high - min_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop_raw = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    chop_12h = np.where(denominator == 0, 50.0, chop_raw)  # Set to 50 when range is zero
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # === 1-day Williams %R (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_raw = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when HH == LL
    williams_1d = np.where((highest_high - lowest_low) == 0, -50.0, williams_raw)
    williams_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_1d)
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(chop_12h_aligned[i]) or 
            np.isnan(williams_1d_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop_val = chop_12h_aligned[i]
        williams_val = williams_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume for confirmation
        
        # === EXIT CONDITIONS ===
        if position == 1:  # Long position
            # Exit when: chop < 38.2 (trending) OR Williams %R > -50 (mean reversion unwinds)
            if chop_val < 38.2 or williams_val > -50:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when: chop < 38.2 (trending) OR Williams %R < -50 (mean reversion unwinds)
            if chop_val < 38.2 or williams_val < -50:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade in ranging markets (Choppiness > 61.8)
            if chop_val > 61.8 and vol_confirm:
                # Long when Williams %R < -80 (oversold)
                if williams_val < -80:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short when Williams %R > -20 (overbought)
                elif williams_val > -20:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Chop61.8_Williams%R_Volume1.5x"
timeframe = "12h"
leverage = 1.0