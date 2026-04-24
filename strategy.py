#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h VWAP-EMA50 trend filter and volume confirmation.
- Long when price breaks above Donchian upper band AND 12h VWAP > 12h EMA50 AND volume > 1.5x 20-period average volume
- Short when price breaks below Donchian lower band AND 12h VWAP < 12h EMA50 AND volume > 1.5x 20-period average volume
- Exit on opposite Donchian breakout or when volume drops below average (loss of momentum)
- Fixed position size 0.25 to minimize fee churn and control drawdown
- Uses 4h primary with 12h HTF for regime + volume confirmation to target 75-150 trades over 4 years
- Donchian provides objective breakout levels; VWAP/EMA50 confirms institutional flow; volume ensures conviction
- Designed to work in both bull (breakouts with volume) and bear (volume-driven breakdowns) markets
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
    
    # Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data ONCE before loop for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h VWAP and EMA50
    vwap_12h = (df_12h['close'] * df_12h['volume']).expanding().sum() / df_12h['volume'].expanding().sum()
    vwap_12h = vwap_12h.values
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Regime: bullish if VWAP > EMA50, bearish if VWAP < EMA50
    bullish_regime = vwap_12h_aligned > ema_50_12h_aligned
    bearish_regime = vwap_12h_aligned < ema_50_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above upper band AND bullish regime AND volume confirmation
            if close[i] > upper[i] and bullish_regime[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND bearish regime AND volume confirmation
            elif close[i] < lower[i] and bearish_regime[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below lower band OR loss of volume confirmation (momentum fading)
            if close[i] < lower[i] or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above upper band OR loss of volume confirmation (momentum fading)
            if close[i] > upper[i] or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hVWAP_EMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0