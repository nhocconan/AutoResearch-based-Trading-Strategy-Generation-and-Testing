#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1d ATR regime filter and volume spike confirmation.
- Long when price breaks above H4 AND 1d ATR(14) > 1d ATR(50) (high volatility regime)
- Short when price breaks below L4 AND 1d ATR(14) > 1d ATR(50) (high volatility regime)
- Volume confirmation: current volume > 1.5 * 24-period average volume (moderate spike)
- Exit on opposite Camarilla level (L4 for long exit, H4 for short exit)
- Uses 12h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla H4/L4 levels provide stronger breakout signals than H3/L3
- ATR regime filter ensures we only trade in high volatility environments
- Volume spike confirms momentum behind breakouts
- Signal size: 0.25 discrete levels to minimize fee churn
- Designed to work in both bull (breakouts with momentum) and bear (breakouts with momentum) markets
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
    
    # Calculate Camarilla levels (based on previous period) on 12h data
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h4 = np.roll(close, 1) + 1.5 * (np.roll(high, 1) - np.roll(low, 1))
    camarilla_l4 = np.roll(close, 1) - 1.5 * (np.roll(high, 1) - np.roll(low, 1))
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) for regime detection
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Regime filter: high volatility when ATR(14) > ATR(50)
    high_vol_regime = atr_14 > atr_50
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float)) > 0.5
    
    # Volume confirmation: volume > 1.5 * 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 24)  # Need Camarilla (1), ATR(50), and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H4 AND high volatility regime AND volume confirmation
            if close[i] > camarilla_h4[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L4 AND high volatility regime AND volume confirmation
            elif close[i] < camarilla_l4[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Camarilla L4 (opposite level)
            if close[i] < camarilla_l4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Camarilla H4 (opposite level)
            if close[i] > camarilla_h4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0