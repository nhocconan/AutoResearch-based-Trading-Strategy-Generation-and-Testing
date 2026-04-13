#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h/1d Camarilla pivot reversal with volume confirmation and chop regime filter.
# Uses Camarilla levels (H4/L4) from daily high/low/close as intraday support/resistance.
# Long when price crosses above L4 with volume spike in choppy market (CHOP > 61.8).
# Short when price crosses below H4 with volume spike in choppy market.
# Works in ranging markets by fading extremes; avoids trends via chop filter.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Calculate 1-day ATR(14) for chop filter
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_1d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if np.isnan(tr_1d[i-14:i+1]).any():
            atr_1d[i] = np.nan
        else:
            atr_1d[i] = np.mean(tr_1d[i-14:i+1])
    
    # Calculate 1-day ADX(14) for chop filter (using DI+ and DI-)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    atr_1d_smooth = np.full(len(tr_1d), np.nan)
    plus_di_smooth = np.full(len(tr_1d), np.nan)
    minus_di_smooth = np.full(len(tr_1d), np.nan)
    
    for i in range(14, len(tr_1d)):
        if np.isnan(atr_1d[i-14:i+1]).any() or np.isnan(plus_dm[i-14:i+1]).any() or np.isnan(minus_dm[i-14:i+1]).any():
            atr_1d_smooth[i] = np.nan
            plus_di_smooth[i] = np.nan
            minus_di_smooth[i] = np.nan
        else:
            atr_1d_smooth[i] = np.mean(tr_1d[i-14:i+1])
            plus_di_smooth[i] = 100 * np.mean(plus_dm[i-14:i+1]) / atr_1d_smooth[i]
            minus_di_smooth[i] = 100 * np.mean(minus_dm[i-14:i+1]) / atr_1d_smooth[i]
    
    # DX and ADX
    dx = np.full(len(tr_1d), np.nan)
    adx_1d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if np.isnan(plus_di_smooth[i]) or np.isnan(minus_di_smooth[i]):
            dx[i] = np.nan
        else:
            dx[i] = 100 * np.abs(plus_di_smooth[i] - minus_di_smooth[i]) / (plus_di_smooth[i] + minus_di_smooth[i])
    
    for i in range(28, len(tr_1d)):  # 2*period for ADX
        if np.isnan(dx[i-14:i+1]).any():
            adx_1d[i] = np.nan
        else:
            adx_1d[i] = np.mean(dx[i-14:i+1])
    
    # Choppiness Index: higher = more choppy (range-bound)
    # CHOP = 100 * log10(sum(atr/14) / log10(highest-high-lowest-low)) / log10(n)
    # Simplified: use ADX < 20 as choppy regime
    choppy_regime = adx_1d < 20  # ADX < 20 indicates ranging market
    
    # Align all indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    choppy_regime_aligned = align_htf_to_ltf(prices, df_1d, choppy_regime.astype(float))
    
    # Calculate average volume (2-period = 1 day) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(2, n):
        if np.isnan(volume[i-2:i]).any():
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-2:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(2, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(choppy_regime_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        is_choppy = choppy_regime_aligned[i] > 0.5  # boolean from aligned float
        
        # Volume confirmation: current volume > 2x average volume (significant spike)
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: price crosses above L4 (support) in choppy market with volume spike
            if (price > l4_level and
                close[i-1] <= l4_level and  # confirm crossover
                is_choppy and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below H4 (resistance) in choppy market with volume spike
            elif (price < h4_level and
                  close[i-1] >= h4_level and  # confirm crossover
                  is_choppy and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below L4 or chop ends
            if (price < l4_level and
                close[i-1] >= l4_level) or not is_choppy:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above H4 or chop ends
            if (price > h4_level and
                close[i-1] <= h4_level) or not is_choppy:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Reversal_Volume_Chop"
timeframe = "12h"
leverage = 1.0