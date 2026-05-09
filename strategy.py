#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-week Relative Strength Index (RSI) extreme readings and 1-day Bollinger Band squeeze.
# In low volatility regimes (BB width < 20th percentile on daily), price tends to mean-revert.
# Uses weekly RSI to identify overbought/oversold conditions: RSI < 30 for long, RSI > 70 for short.
# Entry: Long when weekly RSI < 30, daily BB squeeze, and price crosses above daily 20 SMA.
#        Short when weekly RSI > 70, daily BB squeeze, and price crosses below daily 20 SMA.
# Exit: When volatility regime shifts to high volatility (BB width > 80th percentile) or price reverts to the 20 SMA.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WeeklyRSI_BB_Squeeze"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.fillna(50)  # neutral when no loss
    rsi_14_values = rsi_14.values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_values)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band squeeze: low volatility when BB width < 20th percentile
    bb_width_percentile = bb_width.rolling(window=100, min_periods=100).quantile(0.2)
    bb_squeeze = bb_width < bb_width_percentile
    bb_squeeze_values = bb_squeeze.values
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze_values)
    
    # Bollinger mid-band (20 SMA) for mean reversion target
    bb_mid = sma_20.values
    bb_mid_aligned = align_htf_to_ltf(prices, df_1d, bb_mid)
    
    # Price position relative to BB mid-band
    price_above_mid = close > bb_mid_aligned
    price_below_mid = close < bb_mid_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_aligned[i]) or
            np.isnan(bb_squeeze_aligned[i]) or
            np.isnan(bb_mid_aligned[i]) or
            np.isnan(price_above_mid[i]) or np.isnan(price_below_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly RSI oversold (<30) + daily BB squeeze + price above daily 20 SMA
            if (rsi_14_aligned[i] < 30) and bb_squeeze_aligned[i] and price_above_mid[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly RSI overbought (>70) + daily BB squeeze + price below daily 20 SMA
            elif (rsi_14_aligned[i] > 70) and bb_squeeze_aligned[i] and price_below_mid[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts to high OR price crosses below BB mid
            if (not bb_squeeze_aligned[i]) or (not price_above_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts to high OR price crosses above BB mid
            if (not bb_squeeze_aligned[i]) or (not price_below_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals