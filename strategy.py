#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and Choppiness Index regime filter.
# Long when: Price breaks above R1 with volume > 1.5x 20-period average and CHOP > 61.8 (ranging market)
# Short when: Price breaks below S1 with volume > 1.5x 20-period average and CHOP > 61.8
# Exit when price returns to the 1-day close level or volatility increases.
# Designed for ~25-40 trades/year per symbol.
name = "4h_Camarilla_R1S1_Breakout_Volume_CHOP"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    hl_range = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * hl_range / 12
    camarilla_s1 = close_1d - 1.1 * hl_range / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20 * 1.5)
    
    # Choppiness Index regime filter (CHOP > 61.8 = ranging market)
    # Calculate True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)  # Avoid division by zero
    
    # Regime filter: CHOP > 61.8 (ranging market)
    chop_filter = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ok = volume_filter[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation in ranging market
            if price > r1 and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation in ranging market
            elif price < s1 and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to 1-day close or volatility increases (CHOP < 50)
            if price < close_1d[-1] or chop[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to 1-day close or volatility increases (CHOP < 50)
            if price > close_1d[-1] or chop[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals