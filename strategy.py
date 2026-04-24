#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter, volume spike (>2.0x 24-bar average), and ATR regime filter (current ATR > 0.7x 50-bar average).
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 7-25 trades/year (30-100 total over 4 years) to stay fee-efficient.
- Combines Camarilla structure + 1w trend filter + volume/volatility confirmation.
- Works in bull/bear: trend filter ensures alignment with higher timeframe direction; volume/volatility filters avoid low-conviction entries.
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior 1w OHLC (completed 1w bar)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Align to 1d timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate Camarilla levels
    camarilla_h3 = close_1w_aligned + 1.1 * (high_1w_aligned - low_1w_aligned) / 4
    camarilla_l3 = close_1w_aligned - 1.1 * (high_1w_aligned - low_1w_aligned) / 4
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR(14) for volatility regime filter
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # ATR ratio: current ATR / 50-period average (avoid low volatility chop)
    atr_ma_long = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma_long > 0, atr_ma_long, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, atr_period, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average) + ATR ratio > 0.7 (avoid low vol)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        vol_regime = atr_ratio[i] > 0.7
        
        if position == 0:
            # Long: Close > H3 AND price above 1w EMA50 AND volume confirmation AND vol regime
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND price below 1w EMA50 AND volume confirmation AND vol regime
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm and vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L3 OR price crosses below 1w EMA50
            if close[i] < camarilla_l3[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H3 OR price crosses above 1w EMA50
            if close[i] > camarilla_h3[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_VolumeATR_Filter_v1"
timeframe = "1d"
leverage = 1.0