#!/usr/bin/env python3
"""
Hypothesis: Weekly price extremes act as significant support/resistance levels.
Price tends to reverse or consolidate when reaching prior week's high/low.
Combining weekly range boundaries with daily RSI extremes and volume confirmation
creates high-probability mean-reversion trades. Targets 50-80 trades total over 4 years
by requiring price touch of weekly high/low, RSI <30 or >70, and volume >1.5x average.
Works in both bull (buy dips to weekly support) and bear (sell rallies to weekly resistance).
"""

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
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly high/low from previous week
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: 20-day average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align weekly levels to daily timeframe
    whigh_daily = align_htf_to_ltf(prices, df_1w, whigh)
    wlow_daily = align_htf_to_ltf(prices, df_1w, wlow)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 35  # warmup for RSI and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(whigh_daily[i]) or np.isnan(wlow_daily[i]) or
            np.isnan(rsi_values[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        rsi_val = rsi_values[i]
        
        if position == 0:
            # Long: price at weekly low, oversold RSI, volume confirmation
            if price <= wlow_daily[i] * 1.002 and rsi_val < 30 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price at weekly high, overbought RSI, volume confirmation
            elif price >= whigh_daily[i] * 0.998 and rsi_val > 70 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches weekly high or RSI overbought
            if price >= whigh_daily[i] * 0.995 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches weekly low or RSI oversold
            if price <= wlow_daily[i] * 1.005 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyHighLow_RSI_Volume"
timeframe = "1d"
leverage = 1.0