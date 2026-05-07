#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 12-hour trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA).
# Long when: Bull Power > 0 AND Bear Power < 0 (strong bullish momentum) AND 12h EMA50 rising AND volume > 1.5 * EMA20(volume).
# Short when: Bull Power < 0 AND Bear Power > 0 (strong bearish momentum) AND 12h EMA50 falling AND volume > 1.5 * EMA20(volume).
# Exit when Bull Power and Bear Power converge (|Bull Power| < 0.5 * ATR(14) and |Bear Power| < 0.5 * ATR(14)).
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag and improve generalization.
# Works in bull markets via sustained Bull Power > 0 and in bear markets via sustained Bear Power > 0.
name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: 13-period EMA of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ATR(14) for exit condition
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_rising[1:] = ema_50_12h[1:] > ema_50_12h[:-1]
    ema_50_falling[1:] = ema_50_12h[1:] < ema_50_12h[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr_14[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND EMA50(12h) rising AND volume spike
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Bull Power < 0 AND Bear Power > 0 AND EMA50(12h) falling AND volume spike
            short_condition = (bull_power[i] < 0) and (bear_power[i] > 0) and ema_50_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Weakening bullish momentum (|Bull Power| < 0.5 * ATR and |Bear Power| < 0.5 * ATR)
            if (abs(bull_power[i]) < 0.5 * atr_14[i]) and (abs(bear_power[i]) < 0.5 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Weakening bearish momentum (|Bull Power| < 0.5 * ATR and |Bear Power| < 0.5 * ATR)
            if (abs(bull_power[i]) < 0.5 * atr_14[i]) and (abs(bear_power[i]) < 0.5 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals