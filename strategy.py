#!/usr/bin/env python3
# 6h_1D_MeanReversion_VolatilityExpansion
# Hypothesis: Mean reversion at Bollinger Band extremes (20,2) combined with volatility expansion detection
# using Bollinger Band width percentile. Works in both bull and bear markets by fading extremes during
# high volatility regimes. Targets 15-35 trades per year with strict entry conditions.

name = "6h_1D_MeanReversion_VolatilityExpansion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bollinger Bands (20, 2) on 6h close
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width for volatility regime detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Daily trend filter using EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # High volatility regime: BB width > 70th percentile
        high_vol = bb_width_percentile[i] > 0.70
        
        if position == 0:
            # LONG: Price at lower BB + high volatility + price above daily EMA34 (avoid strong downtrend)
            if (close[i] <= bb_lower[i] and 
                high_vol and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at upper BB + high volatility + price below daily EMA34 (avoid strong uptrend)
            elif (close[i] >= bb_upper[i] and 
                  high_vol and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above middle band OR volatility drops
            if (close[i] >= bb_middle[i] or 
                bb_width_percentile[i] < 0.30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below middle band OR volatility drops
            if (close[i] <= bb_middle[i] or 
                bb_width_percentile[i] < 0.30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals