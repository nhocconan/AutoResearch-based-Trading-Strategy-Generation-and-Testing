# 165158
#!/usr/bin/env python3
"""
1d_Keltner_Channel_RSI_Trend_Filter
Hypothesis: Price breakouts from Keltner Channel (ATR-based volatility bands) on daily timeframe,
with RSI momentum filter and 1-week trend alignment, capture sustained moves in both bull and bear markets.
Keltner Channels adapt to volatility, reducing false breakouts in ranging periods. Weekly trend filter ensures
trades align with higher-timeframe momentum. Designed for low trade frequency (~15-25/year) to minimize fee drag.
"""

name = "1d_Keltner_Channel_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Keltner Channel and RSI (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for Keltner Channel
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channel: EMA(20) ± 2 * ATR
    ema20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + (2 * atr)
    lower_keltner = ema20 - (2 * atr)
    
    # RSI(14)
    delta = pd.Series(df_1d['close']).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 1d timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above upper Keltner, RSI > 50 (bullish momentum), price above weekly EMA34 (uptrend)
            if (close[i] > upper_keltner_aligned[i] and 
                rsi_aligned[i] > 50 and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner, RSI < 50 (bearish momentum), price below weekly EMA34 (downtrend)
            elif (close[i] < lower_keltner_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below lower Keltner (mean reversion) OR RSI < 40 (losing momentum)
            if (close[i] < lower_keltner_aligned[i] or 
                rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above upper Keltner (mean reversion) OR RSI > 60 (losing momentum)
            if (close[i] > upper_keltner_aligned[i] or 
                rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals