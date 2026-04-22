#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Williams %R mean reversion + volume spike
# Long when CHOP > 61.8 (range) + WILLR < -80 (oversold) + volume spike
# Short when CHOP > 61.8 (range) + WILLR > -20 (overbought) + volume spike
# Exit when WILLR crosses back through -50 or volatility drops
# Works in ranging markets (2025-2026 bear/range) by fading extremes; avoids trends where WILLR fails
# Target: 20-30 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # WILLR = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    willr_1d = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    willr_1d = np.where((highest_high - lowest_low) == 0, -50, willr_1d)
    
    # Load 1d data for Choppiness Index calculation
    # CHOP = 100 * log10(SUM(ATR(1)) / (HH - LL)) / log10(N)
    # where ATR(1) = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    atr1_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr1_sum / (hh_14 - ll_14)) / np.log10(14)
    # Handle division by zero when hh == ll
    chop_1d = np.where((hh_14 - ll_14) == 0, 50, chop_1d)
    
    # Align to 4h
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(willr_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        willr = willr_aligned[i]
        chop = chop_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: choppy market (range) + oversold + volume spike
            if chop > 61.8 and willr < -80 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: choppy market (range) + overbought + volume spike
            elif chop > 61.8 and willr > -20 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: WILLR crosses back through -50 or volatility drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when WILLR crosses above -50 (overbought territory) or volume drops
                if willr > -50 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when WILLR crosses below -50 (oversold territory) or volume drops
                if willr < -50 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Choppiness_WilliamsR_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0