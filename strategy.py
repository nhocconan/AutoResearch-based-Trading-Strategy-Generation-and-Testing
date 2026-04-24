#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation.
- Long when price breaks above H3 level AND close > 1d EMA50 (bullish trend)
- Short when price breaks below L3 level AND close < 1d EMA50 (bearish trend)
- Volume must be > 1.5 * median volume of last 20 bars (volume spike filter)
- Exit on opposite Camarilla level touch (L3 for long, H3 for short) for mean reversion in chop
- Uses 12h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla levels provide intraday support/resistance that work in ranging markets
- 1d EMA50 ensures alignment with longer-term trend to avoid whipsaws in trending markets
- Volume spike filter adapts to changing market conditions, reducing false breakouts
- Designed for BTC/ETH with edge in both bull (breakout continuation) and bear (mean reversion at levels) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (H3, L3) using previous period (no look-ahead)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using typical Camarilla calculation based on previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    volume_series = pd.Series(volume)
    volume_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * volume_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 level, trend up (close > EMA50), volume confirmation
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 level, trend down (close < EMA50), volume confirmation
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches or breaks below L3 level (mean reversion)
            if close[i] <= camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches or breaks above H3 level (mean reversion)
            if close[i] >= camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0