# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_1w_RSI_Overbought_Oversold_Simple_v1
Hypothesis: On daily timeframe, buy when weekly RSI(14) is oversold (<30) and daily price is above daily VWAP,
sell when weekly RSI is overbought (>70) and daily price is below daily VWAP. Uses weekly trend filter for alignment.
Designed for 7-25 trades/year by requiring weekly RSI extremes, reducing noise and avoiding whipsaws.
Works in bull markets via buying oversold dips and in bear markets via selling overbought rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Overbought_Oversold_Simple_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Weekly RSI(14)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily trend filter: price above/below daily EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # VWAP filter: price above VWAP for long, below for short
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        # Weekly RSI extremes
        rsi_oversold = rsi_1w_aligned[i] < 30
        rsi_overbought = rsi_1w_aligned[i] > 70
        
        # Entry conditions
        long_entry = rsi_oversold and above_vwap
        short_entry = rsi_overbought and below_vwap
        
        # Exit conditions: RSI returns to neutral zone (40-60) or VWAP breach
        long_exit = rsi_1w_aligned[i] >= 40 or close[i] < vwap[i]
        short_exit = rsi_1w_aligned[i] <= 60 or close[i] > vwap[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals