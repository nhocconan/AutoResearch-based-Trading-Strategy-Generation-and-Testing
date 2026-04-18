#!/usr/bin/env python3
"""
6h_AdaptiveKeltner_Momentum
6h strategy using adaptive Keltner channels with momentum confirmation.
- Uses ATR-based channel width that adapts to volatility regime
- Long: Price touches lower Keltner + momentum > 0 + volume > 1.5x average
- Short: Price touches upper Keltner + momentum < 0 + volume > 1.5x average
- Exit: Opposite touch or momentum reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in ranging markets (mean reversion at Keltner touches) and trending markets (momentum continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for adaptive Keltner channels and momentum
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA(20) on daily close for Keltner middle
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate momentum (ROC 5-period) on daily close
    momentum_1d = np.concatenate([[np.nan, np.nan, np.nan, np.nan, np.nan], 
                                  (close_1d[5:] - close_1d[:-5]) / close_1d[:-5] * 100])
    
    # Calculate volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    momentum_aligned = align_htf_to_ltf(prices, df_1d, momentum_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate adaptive Keltner channels (2.0 * ATR multiplier)
    keltner_upper = ema_20_aligned + 2.0 * atr_14_aligned
    keltner_lower = ema_20_aligned - 2.0 * atr_14_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(momentum_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Momentum conditions
        mom_pos = momentum_aligned[i] > 0
        mom_neg = momentum_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Keltner touch conditions (using close price)
        touch_lower = close[i] <= keltner_lower[i]
        touch_upper = close[i] >= keltner_upper[i]
        
        if position == 0:
            # Long: touch lower Keltner + positive momentum + volume confirmation
            if touch_lower and mom_pos and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: touch upper Keltner + negative momentum + volume confirmation
            elif touch_upper and mom_neg and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: touch upper Keltner or momentum turns negative
            if touch_upper or not mom_pos:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: touch lower Keltner or momentum turns positive
            if touch_lower or mom_pos:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_AdaptiveKeltner_Momentum"
timeframe = "6h"
leverage = 1.0