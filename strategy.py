#!/usr/bin/env python3
# 12h_1d_tema_trend_v1
# Strategy: 12h Tema (Triple Exponential Moving Average) trend with 1d volume filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Tema reduces lag while maintaining smooth trend. Long when 12h close > 12h Tema(30) and 1d volume > 1.5x 20-period average.
# Short when 12h close < 12h Tema(30) and 1d volume > 1.5x 20-period average. Avoids choppy markets with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_tema_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Tema(30) calculation
    # Tema = 3*EMA1 - 3*EMA2 + EMA3 where EMA1 = EMA(close,30), EMA2 = EMA(EMA1,30), EMA3 = EMA(EMA2,30)
    ema1 = pd.Series(close).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema2 = pd.Series(ema1).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema3 = pd.Series(ema2).ewm(span=30, adjust=False, min_periods=30).mean().values
    tema = 3 * ema1 - 3 * ema2 + ema3
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(tema[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend: close vs Tema
        above_tema = close[i] > tema[i]
        below_tema = close[i] < tema[i]
        
        # Entry conditions
        # Long: Price above Tema AND volume confirmation
        if above_tema and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price below Tema AND volume confirmation
        elif below_tema and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses back through Tema (mean reversion signal)
        elif position == 1 and below_tema:
            position = 0
            signals[i] = 0.0
        elif position == -1 and above_tema:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals