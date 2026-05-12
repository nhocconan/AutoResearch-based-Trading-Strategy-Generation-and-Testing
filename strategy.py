#!/usr/bin/env python3
# 6h_Chaikin_Flow_1dTrend_Volume_Weighted_Momentum
# Hypothesis: Chaikin Money Flow (CMF) measures institutional buying/selling pressure. Combined with 1-day EMA trend filter and volume-weighted momentum, it captures sustained moves in both bull and bear markets. CMF > 0 indicates buying pressure, CMF < 0 selling pressure. The 1d EMA ensures alignment with higher timeframe direction. Volume-weighting filters weak signals. Target: 20-40 trades/year.

name = "6h_Chaikin_Flow_1dTrend_Volume_Weighted_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Chaikin Money Flow (20) - measures money flow volume
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period SMA of Money Flow Volume / 20-period SMA of Volume
    hl_range = high - low
    # Avoid division by zero
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    
    # Calculate 20-period averages
    mfv_ma = pd.Series(mfv).rolling(window=20, min_periods=20).mean().values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Avoid division by zero
    cmf = np.divide(mfv_ma, vol_ma, out=np.zeros_like(mfv_ma), where=vol_ma!=0)
    
    # Volume-weighted momentum: price change weighted by volume
    price_change = close - np.roll(close, 1)
    price_change[0] = 0  # First value has no previous
    vol_weighted_momentum = pd.Series(price_change * volume).rolling(window=10, min_periods=10).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cmf[i]) or 
            np.isnan(vol_weighted_momentum[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Positive CMF (buying pressure) + 1d EMA50 uptrend + positive volume-weighted momentum
            if (cmf[i] > 0.05 and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_weighted_momentum[i] > 0):
                signals[i] = 0.25
                position = 1
            # SHORT: Negative CMF (selling pressure) + 1d EMA50 downtrend + negative volume-weighted momentum
            elif (cmf[i] < -0.05 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_weighted_momentum[i] < 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF turns negative or momentum fades
            if cmf[i] < -0.02 or vol_weighted_momentum[i] < -0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF turns positive or momentum fades
            if cmf[i] > 0.02 or vol_weighted_momentum[i] > 0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals