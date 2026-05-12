#/usr/bin/env python3
"""
4h_1d_1w_Trend_Pullback_Entry_v1
Hypothesis: In trending markets (1w EMA34), price pulls back to the 4h EMA21 and resumes trend with volume confirmation. Works in bull/bear via 1w trend filter. Uses EMA pullback for higher win rate than breakouts, reducing false signals. Targets 20-40 trades/year.
"""

name = "4h_1d_1w_Trend_Pullback_Entry_v1"
timeframe = "4h"
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
    
    # 4h EMA21 for pullback entry
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # 1d data for context (optional filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(ema_21[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (price > 1w EMA34), pullback to EMA21, volume confirmation
            if (close[i] > ema_34_1w_aligned[i] and 
                low[i] <= ema_21[i] and  # touched or went below EMA21
                close[i] > ema_21[i] and  # now back above EMA21
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < 1w EMA34), pullback to EMA21, volume confirmation
            elif (close[i] < ema_34_1w_aligned[i] and 
                  high[i] >= ema_21[i] and  # touched or went above EMA21
                  close[i] < ema_21[i] and  # now back below EMA21
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend broken or opposite pullback
            if close[i] < ema_34_1w_aligned[i] or \
               (high[i] >= ema_21[i] and close[i] < ema_21[i]):  # rejected at EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend broken or opposite pullback
            if close[i] > ema_34_1w_aligned[i] or \
               (low[i] <= ema_21[i] and close[i] > ema_21[i]):  # rejected at EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals