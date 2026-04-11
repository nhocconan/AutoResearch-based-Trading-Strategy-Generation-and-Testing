#!/usr/bin/env python3
# 4h_1d_market_facilitator_index_volume_v1
# Strategy: 4h Market Facilitator Index (MFI) with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: MFI combines price and volume to identify buying/selling pressure. 
# In bull markets, MFI > 50 with rising volume confirms accumulation. 
# In bear markets, MFI < 50 with rising volume confirms distribution. 
# Uses 1d EMA50 for trend filter to avoid counter-trend trades. Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_market_facilitator_index_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Typical Price
    typical_price = (high + low + close) / 3.0
    
    # Money Flow
    money_flow = typical_price * volume
    
    # Positive and Negative Money Flow
    positive_flow = np.where(typical_price > np.roll(typical_price, 1), money_flow, 0)
    negative_flow = np.where(typical_price < np.roll(typical_price, 1), money_flow, 0)
    
    # Money Flow Ratio and MFI (14-period)
    pos_sum = pd.Series(positive_flow).rolling(window=14, min_periods=14).sum()
    neg_sum = pd.Series(negative_flow).rolling(window=14, min_periods=14).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(mfi.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: MFI momentum + volume + trend alignment
        if (mfi.iloc[i] > 50 and mfi.iloc[i] > mfi.iloc[i-1] and  # MFI above 50 and rising
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (mfi.iloc[i] < 50 and mfi.iloc[i] < mfi.iloc[i-1] and  # MFI below 50 and falling
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: MFI momentum divergence or trend change
        elif position == 1 and (mfi.iloc[i] <= 50 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (mfi.iloc[i] >= 50 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals