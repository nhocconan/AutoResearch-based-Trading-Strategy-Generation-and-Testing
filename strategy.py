#!/usr/bin/env python3
# 12H_CCI_OVERBOUGHT_OVERSOLD_1D_TREND_FILTER
# Hypothesis: CCI identifies overbought/oversold conditions. In 1d uptrend, go long when CCI crosses above -100 from below; in 1d downtrend, go short when CCI crosses below +100 from above. The 1d trend filter avoids counter-trend trades, while CCI captures mean reversion within the trend. Designed for 12h timeframe to target 15-25 trades/year.

name = "12H_CCI_OVERBOUGHT_OVERSOLD_1D_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily data for CCI calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # CCI(20) calculation
    tp = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # CCI cross signals
    cci_cross_up = (cci_aligned > -100) & (np.roll(cci_aligned, 1) <= -100)
    cci_cross_down = (cci_aligned < 100) & (np.roll(cci_aligned, 1) >= 100)
    cci_cross_up[0] = False
    cci_cross_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(cci_aligned[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + CCI crosses above -100
            if (close[i] > ema34_aligned[i] and cci_cross_up[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + CCI crosses below +100
            elif (close[i] < ema34_aligned[i] and cci_cross_down[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or CCI crosses below +100 (overbought)
            if (close[i] <= ema34_aligned[i] or cci_cross_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or CCI crosses above -100 (oversold)
            if (close[i] >= ema34_aligned[i] or cci_cross_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals