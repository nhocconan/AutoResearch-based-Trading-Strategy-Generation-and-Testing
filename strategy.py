#!/usr/bin/env python3
"""
6h_RSI_Pullback_to_MA_1dTrend
Hypothesis: Uses 60-period EMA on 6h as dynamic support/resistance with RSI(14) pullbacks (RSI<30 for long, RSI>70 for short) in the direction of 1-day EMA34 trend. Works in bull/bear by following trend direction. Targets 15-25 trades/year via strict pullback conditions.
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
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 60-period EMA on 6h for dynamic support/resistance
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for EMA60 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_60[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Pullback conditions: price near EMA60 with RSI extreme
        near_ema = abs(close[i] - ema_60[i]) / ema_60[i] < 0.01  # Within 1% of EMA60
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        long_setup = near_ema and rsi_oversold and uptrend
        short_setup = near_ema and rsi_overbought and downtrend
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        long_exit = rsi[i] > 40
        short_exit = rsi[i] < 60
        
        if long_setup and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_setup and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_RSI_Pullback_to_MA_1dTrend"
timeframe = "6h"
leverage = 1.0