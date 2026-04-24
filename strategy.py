#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation.
- Long when price breaks above weekly R3 level and close > weekly EMA34 (bullish weekly trend)
- Short when price breaks below weekly S3 level and close < weekly EMA34 (bearish weekly trend)
- Volume must be > 2.0x 20-period average for confirmation (strict filter to reduce trades)
- Uses 6h primary timeframe with 1w HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla pivots from weekly data provide institutional support/resistance levels
- Weekly trend filter ensures we only trade with the higher timeframe momentum
- Volume spike confirms institutional participation in breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Camarilla pivots and EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week's OHLC)
    # Camarilla formula: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.55*(high-low)
    # H1 = close + 0.275*(high-low)
    # L1 = close - 0.275*(high-low)
    # L2 = close - 0.55*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    # We focus on H3 (R3) and L3 (S3) for breakouts
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each weekly bar
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w)  # R3 level
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w)  # S3 level
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly Camarilla levels and EMA34 to 6h timeframe
    # Use additional_delay_bars=1 for Camarilla levels as they're based on completed weekly bar
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3, additional_delay_bars=1)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 2.0x 20-period average volume (strict filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3, weekly trend up (close > EMA34), volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, weekly trend down (close < EMA34), volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops back below weekly R3 or weekly trend turns bearish
            if close[i] < camarilla_h3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above weekly S3 or weekly trend turns bullish
            if close[i] > camarilla_l3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1wEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0