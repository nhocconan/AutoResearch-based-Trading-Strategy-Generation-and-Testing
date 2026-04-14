#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d RSI(14) and 1w Williams %R(14) for mean reversion in extreme zones
# RSI < 30 and Williams %R < -80 indicates oversold conditions -> long
# RSI > 70 and Williams %R > -20 indicates overbought conditions -> short
# Works in both bull and bear markets as it captures exhaustion points during extended moves
# Uses 1d RSI for momentum exhaustion and 1w Williams %R for longer-term extremes
# Low trade frequency expected due to strict dual-timeframe extreme conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load 1d data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    rsi_length = 14
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_length, min_periods=rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_length, min_periods=rsi_length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Load 1w data ONCE for Williams %R
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Williams %R(14)
    highest_high = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1w['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 14)  # Need enough for Williams %R
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_aligned[i]
        williams_val = williams_r_aligned[i]
        
        if position == 0:
            # Enter long: RSI oversold AND Williams %R deeply oversold
            if rsi_val < 30 and williams_val < -80:
                position = 1
                signals[i] = position_size
            # Enter short: RSI overbought AND Williams %R deeply overbought
            elif rsi_val > 70 and williams_val > -20:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral OR Williams %R returns from extreme
            if rsi_val >= 50 or williams_val >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral OR Williams %R returns from extreme
            if rsi_val <= 50 or williams_val <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dRSI_1wWilliamsR_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0