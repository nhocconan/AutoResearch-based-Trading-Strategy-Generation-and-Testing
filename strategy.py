#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Chaikin Money Flow (CMF) breakout with 1d EMA34 trend filter
    # CMF > 0 indicates accumulation, CMF < 0 indicates distribution
    # Breakout confirmed when CMF crosses ±0.1 threshold with price breakout
    # EMA34 filter ensures trading in direction of higher timeframe trend
    # Low turnover expected: CMF crossovers are infrequent (< 25/year)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    
    # Avoid division by zero when high == low
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # small epsilon
    
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    
    # 20-period sums
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    
    cmf = mfv_sum / vol_sum  # oscillates around 0
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(cmf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF crosses above +0.1 (accumulation) + price above EMA34 (uptrend)
            if cmf[i] > 0.1 and (i == 100 or cmf[i-1] <= 0.1) and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF crosses below -0.1 (distribution) + price below EMA34 (downtrend)
            elif cmf[i] < -0.1 and (i == 100 or cmf[i-1] >= -0.1) and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: CMF returns to neutral zone (-0.1 to 0.1) or trend reversal
            if position == 1:
                if cmf[i] < 0.1:  # Return to neutral
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if cmf[i] > -0.1:  # Return to neutral
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_CMF_Accumulation_Distribution_1dEMA34_Trend_v1"
timeframe = "4h"
leverage = 1.0