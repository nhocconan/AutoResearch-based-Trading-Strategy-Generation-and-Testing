#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 12h EMA50 trend filter, volume confirmation, and ATR regime filter.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Volume confirmation requires >2.0x 24-period average to ensure conviction.
- ATR regime filter (current ATR > 0.7x 50-period average) avoids low-momentum whipsaws.
- Exits on Camarilla L4/H4 retest or EMA50 trend violation.
- Designed for 20-30 trades/year (80-120 total over 4 years) to stay within fee-efficient range.
- Combines proven elements: Camarilla structure + 12h trend filter + volume/volatility confirmation.
- Targets BTC and ETH primarily; SOL as secondary validation.
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h OHLC (completed 12h bar)
    high_12h = df_12h['high'].shift(1).values
    low_12h = df_12h['low'].shift(1).values
    close_12h = df_12h['close'].shift(1).values
    
    # Align to 4h timeframe
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate Camarilla levels
    camarilla_h4 = close_12h_aligned + 1.1 * (high_12h_aligned - low_12h_aligned) / 2
    camarilla_l4 = close_12h_aligned - 1.1 * (high_12h_aligned - low_12h_aligned) / 2
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average) + ATR ratio > 0.7 (avoid low vol)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        vol_regime = atr_ratio[i] > 0.7
        
        if position == 0:
            # Long: Close > H4 AND price above 12h EMA50 AND volume confirmation AND vol regime
            if close[i] > camarilla_h4[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Close < L4 AND price below 12h EMA50 AND volume confirmation AND vol regime
            elif close[i] < camarilla_l4[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm and vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L4 OR price crosses below 12h EMA50
            if close[i] < camarilla_l4[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H4 OR price crosses above 12h EMA50
            if close[i] > camarilla_h4[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_12hEMA50_VolumeATR_Filter_v1"
timeframe = "4h"
leverage = 1.0