#!/usr/bin/env python3
"""
6h_1d_Chaikin_Momentum_Pullback
Hypothesis: On 6h timeframe, buy pullbacks to the 21-period EMA during strong bullish momentum 
(Chaikin Money Flow > 0.25 on daily) and sell rallies to EMA during bearish momentum (CMF < -0.25).
Uses daily CMF as regime filter and 6x EMA pullback for entry. Works in bull (buy dips) and 
bear (sell rallies) by fading to the trend-defining EMA when institutional flow confirms direction.
Target: 20-50 trades over 4 years (5-12/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Chaikin_Momentum_Pullback"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6H INDICATORS: 21-period EMA ===
    # EMA with proper min_periods
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 1D INDICATOR: Chaikin Money Flow (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CMF for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)  # avoid div by zero
    
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # 20-period CMF
    cmf_20 = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        mfv_sum = np.sum(mfv[:20])
        vol_sum = np.sum(volume_1d[:20])
        cmf_20[19] = mfv_sum / vol_sum if vol_sum != 0 else 0
        for i in range(20, len(volume_1d)):
            mfv_sum = mfv_sum - mfv[i-20] + mfv[i]
            vol_sum = vol_sum - volume_1d[i-20] + volume_1d[i]
            cmf_20[i] = mfv_sum / vol_sum if vol_sum != 0 else 0
    
    # Align CMF to 6h timeframe (wait for daily close)
    cmf_20_aligned = align_htf_to_ltf(prices, df_1d, cmf_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):  # start after EMA warmup
        # Skip if CMF not available
        if np.isnan(cmf_20_aligned[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Bullish regime: CMF > 0.25, buy pullbacks to EMA
        bull_regime = cmf_20_aligned[i] > 0.25
        # Bearish regime: CMF < -0.25, sell rallies to EMA
        bear_regime = cmf_20_aligned[i] < -0.25
        
        # Long signal: price at or below EMA in bull regime
        long_signal = bull_regime and low[i] <= ema_21[i]
        # Short signal: price at or above EMA in bear regime  
        short_signal = bear_regime and high[i] >= ema_21[i]
        
        # Exit conditions: reverse regime or opposing signal
        exit_long = bear_regime or (position == 1 and high[i] >= ema_21[i] * 1.015)  # take profit at 1.5%
        exit_short = bull_regime or (position == -1 and low[i] <= ema_21[i] * 0.985)   # take profit at 1.5%
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals