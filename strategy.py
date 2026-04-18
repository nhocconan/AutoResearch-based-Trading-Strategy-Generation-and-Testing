#!/usr/bin/env python3
"""
4h_ChaikinMoneyFlow_Pullback_Trend
Hypothesis: Chaikin Money Flow (CMF) pullback to trend during strong institutional flow.
Enter long when price pulls back to 20-period EMA during CMF > 0.25 inflow and rising.
Enter short when price pulls back to 20-period EMA during CMF < -0.25 outflow and falling.
Uses 1d trend filter to avoid counter-trend trades. Designed for low frequency (20-40/year)
by requiring strong CMF extremes + pullback confirmation, reducing whipsaws in ranging markets.
Works in bull markets via long pullbacks and bear markets via short pullbacks with trend alignment.
"""

import numpy as np
import pandas as pd
from mkt_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Chaikin Money Flow (20-period)
    # CMF = sum((close - low - (high - close)) / (high - low) * volume) / sum(volume)
    # Simplified: CMF = sum(((close - low) - (high - close)) * volume / (high - low)) / sum(volume)
    # = sum((2*close - high - low) * volume / (high - low)) / sum(volume)
    mfm = ((2 * close - high - low) / (high - low)) * volume
    mfm = np.where((high - low) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum() / \
          pd.Series(volume).rolling(window=20, min_periods=20).sum()
    cmf = cmf.values
    
    # 20-period EMA for pullback entry
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(cmf[i]) or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        cmf_val = cmf[i]
        ema20 = ema_20[i]
        ema34 = ema_34_aligned[i]
        
        if position == 0:
            # Long: CMF strong inflow (>0.25), price pulls back to EMA20, uptrend (price > EMA34)
            if cmf_val > 0.25 and abs(price - ema20) / ema20 < 0.015 and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: CMF strong outflow (<-0.25), price pulls back to EMA20, downtrend (price < EMA34)
            elif cmf_val < -0.25 and abs(price - ema20) / ema20 < 0.015 and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: CMF turns negative OR price breaks above recent high (take profit)
            if cmf_val < 0 or price > ema20 * 1.03:  # 3% profit target or CMF deterioration
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: CMF turns positive OR price breaks below recent low (take profit)
            if cmf_val > 0 or price < ema20 * 0.97:  # 3% profit target or CMF improvement
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ChaikinMoneyFlow_Pullback_Trend"
timeframe = "4h"
leverage = 1.0