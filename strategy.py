#!/usr/bin/env python3
"""
4h_Chaikin_Money_Flow_Trend_Filter_V1
4h strategy using Chaikin Money Flow (CMF) with 200 EMA trend filter and ATR-based volatility filter.
- Long: CMF > 0.05 + close > EMA200 + ATR ratio (ATR14/ATR50) > 0.8
- Short: CMF < -0.05 + close < EMA200 + ATR ratio (ATR14/ATR50) > 0.8
- Exit: Opposite CMF signal or trend reversal
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
CMF captures institutional money flow, EMA200 filters trend direction, ATR ratio ensures sufficient volatility for meaningful moves
Works in bull markets (strong inflows in uptrend) and bear markets (strong outflows in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volatility context
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate ATR for volatility filter (using 4h data)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    # Avoid division by zero
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0.0)
    
    # Calculate Chaikin Money Flow (CMF) - 20 period
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for ATR50 and CMF20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(atr_ratio[i]) or np.isnan(cmf[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Volatility filter - require sufficient volatility for meaningful moves
        vol_filter = atr_ratio[i] > 0.8
        
        # CMF conditions
        cmf_long = cmf[i] > 0.05
        cmf_short = cmf[i] < -0.05
        
        if position == 0:
            # Long: positive CMF + uptrend + volatility
            if cmf_long and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: negative CMF + downtrend + volatility
            elif cmf_short and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: negative CMF or trend reversal
            if cmf_short or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: positive CMF or trend reversal
            if cmf_long or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Chaikin_Money_Flow_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0