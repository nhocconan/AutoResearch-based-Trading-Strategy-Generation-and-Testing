#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Chaikin Money Flow with 12h trend filter and volatility filter
# Chaikin Money Flow (CMF) measures money flow volume over a period.
# We go long when CMF > 0.1 (bullish accumulation) and short when CMF < -0.1 (bearish distribution),
# confirmed by 12h EMA(50) trend direction and low volatility (ATR ratio < 1.2) to avoid whipsaws.
# Designed for low trade frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "4h_CMF_12hTrend_VolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate ATR(14) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period average ATR
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Calculate Chaikin Money Flow (CMF) over 20 periods
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) != 0, mfm, 0.0)
    mfv = mfm * volume
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values / \
          pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(cmf[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        cmf_val = cmf[i]
        vol_filter = atr_ratio[i] < 1.2  # Low volatility filter
        
        if position == 0:
            # Enter long: CMF > 0.1 (accumulation) + uptrend + low volatility
            if (cmf_val > 0.1 and 
                close[i] > ema50_12h_val and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: CMF < -0.1 (distribution) + downtrend + low volatility
            elif (cmf_val < -0.1 and 
                  close[i] < ema50_12h_val and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CMF turns negative OR price breaks below trend
            if cmf_val < 0 or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CMF turns positive OR price breaks above trend
            if cmf_val > 0 or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals