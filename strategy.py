#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1-day trend filter and volume confirmation
# - Williams %R identifies overbought/oversold conditions for mean reversion entries
# - 1-day EMA filter ensures trades align with higher timeframe trend
# - Volume confirmation filters out low-participation false signals
# - Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Target: 80-150 trades over 4 years to balance opportunity with fee minimization

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 14-period Williams %R on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 50-period EMA on daily close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 24-period volume average (equivalent to 1 day on 4h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i]):
            continue
        
        if position == 0:
            # Long: Oversold (%R < -80) in uptrend (price > daily EMA50) with volume
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > vol_ma_24[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Overbought (%R > -20) in downtrend (price < daily EMA50) with volume
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > vol_ma_24[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses back above -50 (momentum shift) or reverse signal
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses back below -50 (momentum shift) or reverse signal
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_MeanReversion_EMA50_Volume"
timeframe = "4h"
leverage = 1.0