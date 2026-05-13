#!/usr/bin/env python3
"""
1d_200EMA_RSI_Pullback_Strategy
Hypothesis: Buy pullbacks to the 200-day EMA with RSI oversold in uptrends, sell bounces to the 200-day EMA with RSI overbought in downtrends. Uses 1-week trend filter and volume confirmation. Designed for low trade frequency (~15/year) with clear trend-following logic that works in both bull and bear markets by aligning with the primary trend.
"""

name = "1d_200EMA_RSI_Pullback_Strategy"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 200-day EMA
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Get 1-week trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if position == 0:
            # LONG: Price near 200EMA (within 1%) with RSI < 30, volume confirmation, and weekly uptrend
            if (close[i] <= ema_200[i] * 1.01 and close[i] >= ema_200[i] * 0.99) and \
               rsi[i] < 30 and volume_confirm[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near 200EMA (within 1%) with RSI > 70, volume confirmation, and weekly downtrend
            elif (close[i] <= ema_200[i] * 1.01 and close[i] >= ema_200[i] * 0.99) and \
                 rsi[i] > 70 and volume_confirm[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above 200EMA or RSI > 70
            if close[i] > ema_200[i] * 1.02 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below 200EMA or RSI < 30
            if close[i] < ema_200[i] * 0.98 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals