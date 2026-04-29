#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA50 AND volume > 1.5x 20-bar avg
# Exit when power values converge (|Bull Power| < 0.1 * ATR AND |Bear Power| < 0.1 * ATR)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing institutional moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Elder Ray measures bull/bear strength relative to EMA13; works in both bull/bear markets by identifying power shifts.
# Volume confirmation filters weak breakouts; 1d EMA50 ensures alignment with higher timeframe trend.

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate ATR(14) for exit condition
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray (using 6h close)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = ema_13 - low   # Bear Power = EMA(13) - Low
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14, 13)  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        ema_13_val = ema_13[i]
        bull = bull_power[i]
        bear = bear_power[i]
        atr_val = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: power convergence (bull/bear power weakening)
            if abs(bull) < 0.1 * atr_val and abs(bear) < 0.1 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: power convergence (bull/bear power weakening)
            if abs(bull) < 0.1 * atr_val and abs(bear) < 0.1 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume confirmation
            if bull > 0 and bear < 0 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA50 AND volume confirmation
            elif bear > 0 and bull < 0 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals