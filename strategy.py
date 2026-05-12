# 12h 1d EMA Trend with Volume Confirmation
# Hypothesis: 12-hour price crossing above/below 1-day EMA34 with volume surge
# captures medium-term trend changes in both bull and bear markets.
# Low trade frequency (~20/year) minimizes fee drag; EMA34 adapts to volatility.
# Volume confirmation ensures breakouts have institutional participation.
# Works in bull markets (riding trends) and bear markets (catching reversals).
# Uses 1-day EMA as dynamic support/resistance to avoid whipsaws.

name = "12h_1dEMA34_VolumeTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day EMA34 (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False).values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above EMA34 with volume spike
            if close[i] > ema_34_1d_aligned[i] and close[i-1] <= ema_34_1d_aligned[i-1] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below EMA34 with volume spike
            elif close[i] < ema_34_1d_aligned[i] and close[i-1] >= ema_34_1d_aligned[i-1] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses back below EMA34
            if close[i] < ema_34_1d_aligned[i] and close[i-1] >= ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above EMA34
            if close[i] > ema_34_1d_aligned[i] and close[i-1] <= ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals