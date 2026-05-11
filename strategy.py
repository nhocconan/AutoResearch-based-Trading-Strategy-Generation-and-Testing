#!/usr/bin/env python3
name = "6h_ChaikinMoneyFlow_1dTrend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Chaikin Money Flow (CMF) on 6h: 20-period
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = np.where(high == low, 0, mf_multiplier)
    mf_volume = mf_multiplier * volume
    cmf = np.nancumsum(mf_volume) / np.nancumsum(volume)
    cmf = pd.Series(cmf).rolling(window=20, min_periods=20).mean().values
    
    # Align CMF (no additional delay needed)
    # Note: We align the CMF values to 6b timeframe
    # But CMF is already calculated on 6h, so we can use directly
    # Actually, we need to align nothing since it's LTF
    
    # Position sizing
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for CMF (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA data invalid
        if np.isnan(ema34_1d_aligned[i]):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # CMF value at current bar
        cmf_val = cmf[i]
        
        # Trend filter: price vs 1d EMA34
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: CMF turns positive (>0.1) AND price above 1d EMA (uptrend)
            if cmf_val > 0.1 and price_above_ema1d:
                signals[i] = position_size
                position = 1
            # Short: CMF turns negative (<-0.1) AND price below 1d EMA (downtrend)
            elif cmf_val < -0.1 and price_below_ema1d:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: CMF crosses back towards zero (loss of momentum)
            if position == 1:
                if cmf_val < 0.0:  # CMF crossed below zero
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if cmf_val > 0.0:  # CMF crossed above zero
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals