#!/usr/bin/env python3
"""
12h_atr_breakout_v1
Hypothesis: 12-hour ATR-based breakout with daily trend filter to capture trend continuation while avoiding whipsaw.
- Long when price breaks above ATR-based upper band and daily trend is bullish
- Short when price breaks below ATR-based lower band and daily trend is bearish
- Exit on opposite breakout or trend reversal
- Uses volatility-adjusted breakouts to adapt to changing market conditions
- Target: 20-40 trades/year to stay within optimal range for 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_atr_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ATR(14) for volatility-based bands
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate upper and lower bands (ATR multiplier = 1.5)
    ma = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_band = ma + 1.5 * atr
    lower_band = ma - 1.5 * atr
    
    # Daily trend filter using EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_trend = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(daily_trend[i]) or np.isnan(ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price breaks below lower band or daily trend turns bearish
            if close[i] < lower_band[i] or daily_close[i] < daily_ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above upper band or daily trend turns bullish
            if close[i] > upper_band[i] or daily_close[i] > daily_ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price breaks above upper band and daily trend is bullish
            if close[i] > upper_band[i] and daily_close[i] > daily_ema50[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band and daily trend is bearish
            elif close[i] < lower_band[i] and daily_close[i] < daily_ema50[i]:
                position = -1
                signals[i] = -0.25
    
    return signals