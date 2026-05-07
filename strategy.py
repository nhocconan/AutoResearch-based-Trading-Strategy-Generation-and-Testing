# 4h_TRIX_VolumeSpike_Regime
# TRIX (12) zero-cross + volume spike + choppiness regime filter
# Long: TRIX crosses above 0, volume > 2x MA20, CHOP > 61.8 (range)
# Short: TRIX crosses below 0, volume > 2x MA20, CHOP > 61.8 (range)
# Exit: Opposite TRIX cross
# Uses 1d trend filter only for regime context (not entry)
# Target: 20-40 trades/year to avoid fee drag
# Works in sideways markets where TRIX captures momentum shifts in ranges

#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX calculation: EMA(EMA(EMA(close, 12), 12), 12) - then ROC
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change(periods=1) * 100  # ROC of triple EMA
    trix_values = trix.values
    
    # Choppiness Index on 1d: CHOP = 100 * log10(sum(ATR(14)) / (n * log(n)))
    # Simplified: use rolling range / ATR
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    # True range sum over 14 periods
    tr_sum = tr.rolling(window=14, min_periods=14).sum()
    # Max high - min low over 14 periods
    max_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(tr_sum / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX zero-cross signals
        trix_cross_up = trix_values[i-1] <= 0 and trix_values[i] > 0
        trix_cross_down = trix_values[i-1] >= 0 and trix_values[i] < 0
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        chop_condition = chop_aligned[i] > 61.8  # Range regime
        
        if position == 0:
            # Long: TRIX crosses above 0 in range with volume spike
            if trix_cross_up and vol_condition and chop_condition:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0 in range with volume spike
            elif trix_cross_down and vol_condition and chop_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below 0
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above 0
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (triple EMA ROC) captures momentum shifts in ranging markets
# - TRIX crosses above/below zero indicate momentum shifts
# - Volume confirmation (2x average) filters weak signals
# - Choppiness regime filter (CHOP > 61.8) ensures we only trade in ranging markets
# - Works in both bull and bear markets as long as price is ranging
# - Exit on opposite TRIX cross to capture full momentum move
# - Position size 0.25 limits risk and reduces trade frequency
# - Target: 20-40 trades/year to stay within fee drag limits
# - Avoids trending markets where whipsaws occur (CHOP < 38.2 would be avoided)