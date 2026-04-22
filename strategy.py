#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and daily volatility regime filter.
# Uses daily ATR ratio (ATR(7)/ATR(30)) to detect volatility expansion/contraction.
# Volatility expansion (ratio > 1.5) triggers breakout trades; contraction (< 1.2) triggers mean reversion.
# Designed to capture volatility breakouts in trends and mean reversion in low volatility regimes.
# Targets 20-40 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for volatility regime (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(7) and ATR(30)
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    vol_ratio = atr7 / atr30
    vol_ratio = np.where(atr30 == 0, 1.0, vol_ratio)  # Avoid division by zero
    
    # Align volatility ratio to 4h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vol_ratio_val = vol_ratio_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Volatility expansion: breakout strategy
            if vol_ratio_val > 1.5:
                if price > upper and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif price < lower and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Volatility contraction: mean reversion at channel boundaries
            elif vol_ratio_val < 1.2:
                if price <= lower and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif price >= upper and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on retracement to middle or opposite band touch
                if price < mid or price >= upper:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on retracement to middle or opposite band touch
                if price > mid or price <= lower:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VolRatio_Donchian_Breakout_MeanRev"
timeframe = "4h"
leverage = 1.0