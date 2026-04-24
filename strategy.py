#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and ATR volatility regime.
- Long when Williams %R crosses above -80 (oversold reversal) AND close > 1d EMA50 (bullish trend) AND ATR(14) > ATR(50) * 0.8 (vol not too low)
- Short when Williams %R crosses below -20 (overbought reversal) AND close < 1d EMA50 (bearish trend) AND ATR(14) > ATR(50) * 0.8
- Exit on opposite Williams %R cross or trend reversal (close crosses 1d EMA50)
- Uses 12h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Williams %R captures mean reversion extremes that work in both bull and bear markets
- 1d EMA50 ensures alignment with higher timeframe trend to avoid whipsaws
- ATR regime filter avoids low-volatility choppy markets where reversals fail
- Designed for BTC/ETH with edge in ranging markets (mean reversion at extremes) and trending markets (continuation after pullback)
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
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR regime filter: ATR(14) > ATR(50) * 0.8 (avoid low volatility chop)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_14 > (atr_50 * 0.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below), trend up, ATR regime OK
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_1d_aligned[i] and atr_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above), trend down, ATR regime OK
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_1d_aligned[i] and atr_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 OR trend reversal (close < EMA50)
            if (williams_r[i] < -20 and williams_r[i-1] >= -20) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 OR trend reversal (close > EMA50)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_ATRRegime_v1"
timeframe = "12h"
leverage = 1.0