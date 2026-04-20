#!/usr/bin/env python3
# 1d_1w_Chaikin_Momentum_Filter_v1
# Hypothesis: Chaikin Money Flow (CMF) on daily timeframe measures institutional accumulation/distribution.
# Combined with weekly trend filter (EMA34) and price position relative to weekly VWAP to catch
# sustainable moves with institutional backing. Works in bull (accumulation + uptrend) and bear
# (distribution + downtrend) regimes. Target: 15-25 trades per year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Chaikin_Momentum_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter and VWAP
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === Weekly: EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Weekly: VWAP (20-period) ===
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    volume_1w = df_1w['volume'].values
    vwap_num = (typical_price_1w * volume_1w).rolling(window=20, min_periods=20).sum()
    vwap_den = volume_1w.rolling(window=20, min_periods=20).sum()
    vwap_1w = vwap_num / vwap_den
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # === Daily: Chaikin Money Flow (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = ((close - low) - (high - close)) / np.where((high - low) != 0, (high - low), 1.0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    cmf_num = pd.Series(mfv).rolling(window=20, min_periods=20).sum()
    cmf_den = pd.Series(volume).rolling(window=20, min_periods=20).sum()
    cmf = cmf_num / cmf_den
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after weekly EMA warmup
        # Get values
        close_val = close[i]
        cmf_val = cmf[i]
        ema34_val = ema34_1w_aligned[i]
        vwap_val = vwap_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(cmf_val) or np.isnan(ema34_val) or np.isnan(vwap_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF > 0 (accumulation) + price above weekly EMA34 + price above weekly VWAP
            if cmf_val > 0.05 and close_val > ema34_val and close_val > vwap_val:
                signals[i] = 0.25
                position = 1
            # Short: CMF < 0 (distribution) + price below weekly EMA34 + price below weekly VWAP
            elif cmf_val < -0.05 and close_val < ema34_val and close_val < vwap_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF turns negative (distribution) OR price breaks below weekly EMA34
            if cmf_val < -0.05 or close_val < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF turns positive (accumulation) OR price breaks above weekly EMA34
            if cmf_val > 0.05 or close_val > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals