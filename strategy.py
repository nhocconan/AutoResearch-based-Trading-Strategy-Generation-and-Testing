#!/usr/bin/env python3
"""
1d_1w_Keltner_Breakout_Trend_Filter_v1
Hypothesis: Use weekly Keltner channels with ADX trend filter on daily timeframe.
Buy when price breaks above upper Keltner band with ADX > 25 (strong trend).
Sell when price breaks below lower Keltner band with ADX > 25 (strong trend).
Exit when price crosses opposite Keltner band or ADX falls below 20 (weakening trend).
Designed for low trade frequency (<100 total over 4 years) to minimize fee drift.
Works in both bull and bear markets by following strong trends only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Keltner_Breakout_Trend_Filter_v1"
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
    
    # Weekly data for Keltner channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly ATR for Keltner channels
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    # Weekly EMA for Keltner center
    ema_center = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands (2 * ATR)
    upper = ema_center + 2 * atr
    lower = ema_center - 2 * atr
    
    # Align Keltner bands to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    center_aligned = align_htf_to_ltf(prices, df_1w, ema_center)
    
    # Daily ADX for trend filter
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_adx = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr_adx)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr_adx)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend condition
        strong_trend = adx[i] > 25
        weakening_trend = adx[i] < 20
        
        # Breakout conditions with trend filter
        long_breakout = close[i] > upper_aligned[i] and strong_trend
        short_breakout = close[i] < lower_aligned[i] and strong_trend
        
        # Exit conditions
        long_exit = (close[i] < center_aligned[i]) or weakening_trend
        short_exit = (close[i] > center_aligned[i]) or weakening_trend
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals