#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period RSI for mean reversion
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, 0)  # Insert 0 at start to maintain length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 20-period Bollinger Bands
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(upper_bb_val) or np.isnan(lower_bb_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price touches lower Bollinger Band
            if rsi_val < 30 and close_val <= lower_bb_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price touches upper Bollinger Band
            elif rsi_val > 70 and close_val >= upper_bb_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or price touches upper Bollinger Band
            if rsi_val > 50 or close_val >= upper_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or price touches lower Bollinger Band
            if rsi_val < 50 or close_val <= lower_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_DailyRSI_Bollinger_MeanReversion_Session_v1
# Uses daily RSI(14) for overbought/oversold signals
# Uses daily Bollinger Bands(20,2) for entry confirmation
# Enters long when RSI < 30 and price touches lower BB
# Enters short when RSI > 70 and price touches upper BB
# Exits when RSI crosses 50 or price touches opposite BB
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 6h timeframe with ~15-30 trades/year
name = "6h_DailyRSI_Bollinger_MeanReversion_Session_v1"
timeframe = "6h"
leverage = 1.0