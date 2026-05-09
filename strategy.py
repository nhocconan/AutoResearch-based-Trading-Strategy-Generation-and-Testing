#!/usr/bin/env python3
# Hypothesis: 1h timeframe with 4h/1d trend filters and volume confirmation to reduce false breakouts
# Strategy uses 4h RSI for momentum direction, 1d ADX for trend strength, and volume spike for confirmation
# Entry: Long when 4h RSI > 55, 1d ADX > 25, and volume > 1.5x 20-period average
# Entry: Short when 4h RSI < 45, 1d ADX > 25, and volume > 1.5x 20-period average
# Exit: Reverse signal or trailing stop via price action (close below/above prior swing)
# Position size: 0.20 to manage risk in volatile 1h timeframe
# Target: 60-150 total trades over 4 years (15-37/year) with strict entry conditions

name = "1h_RSI_ADX_Volume_Trend_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    delta = pd.Series(df_4h['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM
    plus_di_1d = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr_1d)
    dx = (abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)) * 100
    adx_1d = dx.rolling(window=14, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h RSI > 55 (bullish momentum), 1d ADX > 25 (strong trend), volume spike
            if (rsi_4h_aligned[i] > 55 and 
                adx_1d_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h RSI < 45 (bearish momentum), 1d ADX > 25 (strong trend), volume spike
            elif (rsi_4h_aligned[i] < 45 and 
                  adx_1d_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI drops below 50 (momentum loss) or reverse signal
            if rsi_4h_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI rises above 50 (momentum loss) or reverse signal
            if rsi_4h_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals