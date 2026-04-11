#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with 1d volume confirmation and volatility filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3, H3) act as strong support/resistance. A break above H3 or below L3 with increased volume signals institutional participation. Volatility filter (ATR ratio) avoids choppy markets. Designed for low trade frequency (~20-40/year) to minimize fee drift. Works in bull markets via breakout continuation and bear markets via breakdown continuation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    # Based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Typical price for pivot calculation
    typical_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4, H3, L3, L4
    # H3 = close + 1.1 * range / 2
    # L3 = close - 1.1 * range / 2
    camarilla_h3 = close_1d + 1.1 * range_1d / 2
    camarilla_l3 = close_1d - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # 4h ATR(14) for volatility filter
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_avg_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_avg_50 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]) or \
           np.isnan(atr_ratio[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Volatility filter: avoid extreme volatility (ratio > 3.0) and too low volatility (ratio < 0.5)
        vol_filter = (atr_ratio[i] >= 0.5) and (atr_ratio[i] <= 3.0)
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions
        # Long: Break above H3 with volume confirmation and volatility filter
        if breakout_up and vol_confirm and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Break below L3 with volume confirmation and volatility filter
        elif breakout_down and vol_confirm and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to pivot level (mean reversion)
        elif position == 1 and close[i] <= camarilla_h3_aligned[i] * 0.995:  # Slight buffer
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= camarilla_l3_aligned[i] * 1.005:  # Slight buffer
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals