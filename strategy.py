#!/usr/bin/env python3
"""
6h_RVOL_Pullback_To_VWAP
Hypothesis: On 6h timeframe, enter long when price pulls back to VWAP during high relative volume in uptrend,
and short when price rallies to VWAP during high relative volume in downtrend.
Uses 1d ADX for trend filter and 6h VWAP as dynamic support/resistance.
Target: 20-30 trades/year to minimize fee fade while capturing mean-reversion within trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily ADX for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first bar has no previous close
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Relative volume: current volume / 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 34)  # Warmup for ADX and VWAP
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(vwap[i]) or
            np.isnan(rvol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1d_aligned[i]
        vwap_val = vwap[i]
        rvol_val = rvol[i]
        
        if position == 0:
            # Long: pullback to VWAP with high RVOL in uptrend (ADX > 25)
            if (abs(price - vwap_val) < 0.005 * price and  # within 0.5% of VWAP
                low[i] <= vwap_val <= high[i] and           # price touched VWAP
                rvol_val > 1.5 and                          # volume spike
                adx_val > 25):                              # strong trend
                signals[i] = 0.25
                position = 1
            # Short: rally to VWAP with high RVOL in downtrend (ADX > 25)
            elif (abs(price - vwap_val) < 0.005 * price and  # within 0.5% of VWAP
                  low[i] <= vwap_val <= high[i] and          # price touched VWAP
                  rvol_val > 1.5 and                         # volume spike
                  adx_val > 25):                             # strong trend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price moves 1.5% away from VWAP or trend weakens
            if (abs(price - vwap_val) > 0.015 * price or
                adx_val < 20):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price moves 1.5% away from VWAP or trend weakens
            if (abs(price - vwap_val) > 0.015 * price or
                adx_val < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RVOL_Pullback_To_VWAP"
timeframe = "6h"
leverage = 1.0