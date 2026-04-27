#!/usr/bin/env python3
"""
6h_Keltner_Breakout_ATR_Regime
Hypothesis: Keltner Channel breakout with ATR regime filter and volume confirmation captures strong trends while avoiding whipsaws. 
In bull markets, upper band breakouts trigger longs; in bear markets, lower band breakouts trigger shorts. 
ATR regime filter ensures trades occur only in sufficient volatility environments. 
Volume confirmation adds conviction to breakouts. Targets 12-37 trades/year on 6h to minimize fee drag.
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Keltner Channel (20, 2.0) on 6h timeframe
    atr_period = 20
    ma_period = 20
    multiplier = 2.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Middle line (EMA)
    ema_middle = pd.Series(close).ewm(span=ma_period, adjust=False, min_periods=ma_period).mean().values
    
    # Upper and Lower bands
    upper_band = ema_middle + (multiplier * atr)
    lower_band = ema_middle - (multiplier * atr)
    
    # ATR regime filter: current ATR > 50-day average ATR (from 1d data)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    # 50-day average ATR
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_1d > atr_ma_50  # High volatility regime
    
    # Align ATR regime to 6h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ATR, EMA, and volume MA
    start_idx = max(atr_period, ma_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_middle[i]) or np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        atr_regime_val = atr_regime_aligned[i] > 0.5
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: price breaks above upper band with high volatility regime and volume confirmation
            if close[i] > upper_band[i] and atr_regime_val and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with high volatility regime and volume confirmation
            elif close[i] < lower_band[i] and atr_regime_val and vol_confirm_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or volatility regime ends
            if close[i] < ema_middle[i] or not atr_regime_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle band or volatility regime ends
            if close[i] > ema_middle[i] or not atr_regime_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Keltner_Breakout_ATR_Regime"
timeframe = "6h"
leverage = 1.0