#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Bollinger Band breakout with 1-day RSI filter and volume confirmation
# Long when price breaks above weekly upper BB with RSI(1d) > 50 and volume > 1.5x average
# Short when price breaks below weekly lower BB with RSI(1d) < 50 and volume > 1.5x average
# Weekly Bollinger Bands provide dynamic support/resistance, RSI filters for momentum direction,
# Volume confirms breakout strength. Works in bull/bear markets by capturing genuine momentum shifts.
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing.

name = "1d_1wBB_20_2.0_RSI_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week Bollinger Bands ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Bollinger Bands (20-period, 2 std dev)
    weekly_close = df_1w['close']
    bb_middle = weekly_close.rolling(window=20, min_periods=20).mean().values
    bb_std = weekly_close.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Align weekly Bollinger Bands to daily timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    # Calculate 1-day RSI for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly upper BB with RSI > 50 and volume confirmation
            if close[i] > bb_upper_aligned[i] and rsi_values[i] > 50 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly lower BB with RSI < 50 and volume confirmation
            elif close[i] < bb_lower_aligned[i] and rsi_values[i] < 50 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly middle BB (mean reversion)
            if close[i] < bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly middle BB (mean reversion)
            if close[i] > bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals