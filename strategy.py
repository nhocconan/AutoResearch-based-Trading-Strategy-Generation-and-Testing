#!/usr/bin/env python3
"""
Hypothesis: 4-hour RSI mean reversion with 1-day Bollinger Bands and volume confirmation.
Long when RSI < 30, price touches lower BB, and volume > 1.5x average.
Short when RSI > 70, price touches upper BB, and volume > 1.5x average.
Exit when RSI returns to 50 or price crosses middle band.
Designed for low trade frequency (~20-40/year) to capture mean reversion in ranging markets.
Works in both bull and bear markets by using volatility-adjusted bands.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20,2)
    close_1d = df_1d['close'].values
    ma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = ma20 + (2 * std20)
    lower_bb = ma20 - (2 * std20)
    middle_bb = ma20
    
    # Align BB to lower timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb.values)
    
    # Calculate 4-hour RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(rsi_values[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(middle_bb_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_values[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        middle_bb_val = middle_bb_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: RSI oversold, price at lower BB, volume confirmation
            if (rsi_val < 30 and price <= lower_bb_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought, price at upper BB, volume confirmation
            elif (rsi_val > 70 and price >= upper_bb_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI returns to 50 or price crosses above middle BB
                if (rsi_val >= 50 or price >= middle_bb_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI returns to 50 or price crosses below middle BB
                if (rsi_val <= 50 or price <= middle_bb_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_RSI_BB_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0