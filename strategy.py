#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for trend strength filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian(20) breakout levels on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        rsi_val = rsi_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(rsi_val) or np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h price breaks above upper Donchian band with strong 1d RSI (>50) and sufficient volatility
            if close_val > upper_val and rsi_val > 50 and atr_1d_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: 4h price breaks below lower Donchian band with weak 1d RSI (<50) and sufficient volatility
            elif close_val < lower_val and rsi_val < 50 and atr_1d_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower Donchian band or volatility drops
            if close_val < lower_val or atr_1d_val < 0.5 * atr_1d_val:  # volatility collapse
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian band or volatility drops
            if close_val > upper_val or atr_1d_val < 0.5 * atr_1d_val:  # volatility collapse
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_1dRSI_DonchianBreakout_VolFilter_V1
# Uses 1d RSI(14) as trend strength filter (long when RSI>50, short when RSI<50)
# Enters on 4h Donchian(20) breakouts in direction of 1d trend
# Filters out low volatility periods using 1d ATR
# Exits on opposite band touch or volatility collapse
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_1dRSI_DonchianBreakout_VolFilter_V1"
timeframe = "4h"
leverage = 1.0