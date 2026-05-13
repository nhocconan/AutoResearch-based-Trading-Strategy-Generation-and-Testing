#!/usr/bin/env python3
"""
4h_Chaikin_Money_Flow_Trend_Filter
Hypothesis: Chaikin Money Flow (CMF) measures institutional buying/selling pressure. 
In bull markets: CMF > 0.10 + price > EMA50 = long. 
In bear markets: CMF < -0.10 + price < EMA50 = short.
Uses 12h trend filter for higher timeframe bias and volume confirmation to avoid false signals.
Target: 20-40 trades/year per symbol with controlled risk.
"""

name = "4h_Chaikin_Money_Flow_Trend_Filter"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of volume
    mfv_sum = np.zeros(n)
    vol_sum = np.zeros(n)
    for i in range(20, n):
        mfv_sum[i] = np.sum(mfv[i-20:i])
        vol_sum[i] = np.sum(volume[i-20:i])
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0.0)
    
    # EMA50 for trend
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        cmf_val = cmf[i]
        price_above_ema = close[i] > ema_50[i]
        price_below_ema = close[i] < ema_50[i]
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: CMF > 0.10, price above EMA50, 12h uptrend, volume confirmation
            if cmf_val > 0.10 and price_above_ema and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.10, price below EMA50, 12h downtrend, volume confirmation
            elif cmf_val < -0.10 and price_below_ema and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF < 0.00 or price below EMA50
            if cmf_val < 0.00 or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF > 0.00 or price above EMA50
            if cmf_val > 0.00 or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals