#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike.
- Camarilla pivot levels (R3/S3) identified from prior 1d session provide institutional support/resistance.
- Breakout beyond R3 (bullish) or below S3 (bearish) with volume confirmation captures strong momentum.
- 1d EMA34 ensures alignment with higher-timeframe trend to avoid counter-trend trades.
- Volume spike (>2x 20-period average) confirms institutional participation.
- Position size 0.25 balances profit potential with drawdown control in volatile crypto markets.
- Target trades: 100-180 total over 4 years (25-45/year) to minimize fee drag while capturing sufficient opportunities.
- Works in both bull and bear markets via 1d trend filter and volatility expansion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d session
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But we use R3 and S3: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * camarilla_range
    s3_level = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only enter on breakout with volume confirmation and trend alignment
            if volume_confirm:
                # Long: break above R3 + price above 1d EMA34 (bullish alignment)
                if close[i] > r3_aligned[i] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S3 + price below 1d EMA34 (bearish alignment)
                elif close[i] < s3_aligned[i] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below S3 (reversal below support) OR below EMA34 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 (reversal above resistance) OR above EMA34 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0