#!/usr/bin/env python3
# Hypothesis: 4h 4-period RSI with 1-day Bollinger Band mean reversion. Uses RSI to detect oversold/overbought conditions on the 4h chart, filtered by 1-day Bollinger Bands to ensure trades occur near extreme price levels. This combines short-term momentum exhaustion with longer-term volatility bands to capture reversals in both trending and ranging markets. Designed for low trade frequency (<30/year) to minimize fee drag.

name = "4h_RSI4_BB1D_MeanReversion"
timeframe = "4h"
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
    
    # Calculate 4-period RSI on 4h chart
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = loss.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Get 1-day data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period SMA and standard deviation for Bollinger Bands on 1d
    close_1d_series = pd.Series(close_1d)
    sma20_1d = close_1d_series.rolling(window=20, min_periods=20).mean()
    std20_1d = close_1d_series.rolling(window=20, min_periods=20).std()
    upper_bb_1d = sma20_1d + (2 * std20_1d)
    lower_bb_1d = sma20_1d - (2 * std20_1d)
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_20.values)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_20.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Bollinger Bands
        if np.isnan(rsi[i]) or np.isnan(upper_bb_1d_aligned[i]) or np.isnan(lower_bb_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI oversold (<30) and price near lower Bollinger Band on 1d
            if rsi[i] < 30 and close[i] <= lower_bb_1d_aligned[i] * 1.01:  # Allow small tolerance
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) and price near upper Bollinger Band on 1d
            elif rsi[i] > 70 and close[i] >= upper_bb_1d_aligned[i] * 0.99:  # Allow small tolerance
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>50) or price reaches middle band
            if rsi[i] > 50 or close[i] >= sma20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<50) or price reaches middle band
            if rsi[i] < 50 or close[i] <= sma20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals