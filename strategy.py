#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily VWAP reversion and volume confirmation.
# In mean-reverting markets, price tends to revert to the daily VWAP after deviations.
# Enters long when price crosses below daily VWAP with volume confirmation, short when above.
# Uses 4-hour RSI to filter overbought/oversold conditions.
# Exits when price returns to daily VWAP or RSI reaches neutral.
# Target: 80-160 total trades over 4 years (20-40/year) with size 0.25.

name = "4h_DailyVWAP_Reversion"
timeframe = "4h"
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
    
    # Calculate daily VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Typical price for VWAP calculation
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # 4-hour RSI (14-period) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume ratio: current volume / 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma
    vol_ratio_values = vol_ratio.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or
            np.isnan(rsi_values[i]) or
            np.isnan(vol_ratio_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price below VWAP + volume confirmation + RSI not overbought
            if close[i] < vwap_1d_aligned[i] and vol_ratio_values[i] > 1.2 and rsi_values[i] < 70:
                signals[i] = 0.25
                position = 1
            # Enter short: price above VWAP + volume confirmation + RSI not oversold
            elif close[i] > vwap_1d_aligned[i] and vol_ratio_values[i] > 1.2 and rsi_values[i] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or RSI overbought
            if close[i] >= vwap_1d_aligned[i] or rsi_values[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP or RSI oversold
            if close[i] <= vwap_1d_aligned[i] or rsi_values[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals