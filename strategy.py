#!/usr/bin/env python3
"""
6h_Momentum_Reversal_With_1dTrend
Hypothesis: In 6-hour timeframe, price reverses from overbought/oversold conditions (RSI) when confirmed by 1-day trend (EMA34).
Long when RSI < 30 and price above 1-day EMA34 (oversold in uptrend).
Short when RSI > 70 and price below 1-day EMA34 (overbought in downtrend).
Exit when RSI returns to neutral zone (40-60) or trend reverses.
Designed for 15-30 trades/year to minimize fee dust while capturing mean-reversion moves in both bull and bear markets.
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
    
    # RSI(14) - momentum oscillator
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 14)  # Warmup for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema34 = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) and price above 1-day EMA (uptrend)
            if rsi_val < 30 and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) and price below 1-day EMA (downtrend)
            elif rsi_val > 70 and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI returns to neutral (40-60) or trend turns down
            if 40 <= rsi_val <= 60:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI returns to neutral (40-60) or trend turns up
            if 40 <= rsi_val <= 60:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Momentum_Reversal_With_1dTrend"
timeframe = "6h"
leverage = 1.0