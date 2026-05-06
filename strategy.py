#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND close > 1d EMA50 (uptrend) AND volume > 1.5 * 20-bar avg volume
# Short when Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND close < 1d EMA50 (downtrend) AND volume > 1.5 * 20-bar avg volume
# Exit when Bull Power and Bear Power converge (|Bull Power - Bear Power| < 0.1 * ATR(14)) indicating weakening momentum
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear power relative to EMA13, effective in both trending and ranging markets
# Volume confirmation reduces false signals
# 1d EMA50 provides higher timeframe trend filter for better regime adaptation

name = "6h_ElderRay_1dEMA50_Volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) for exit condition
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Elder Ray signals with trend and volume filters
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND uptrend AND volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND downtrend AND volume spike
            elif bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Elder Ray convergence (weakening bullish momentum)
            if np.abs(bull_power[i] - bear_power[i]) < 0.1 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray convergence (weakening bearish momentum)
            if np.abs(bull_power[i] - bear_power[i]) < 0.1 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals