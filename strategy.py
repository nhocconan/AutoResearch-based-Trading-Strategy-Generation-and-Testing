#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from weekly chart to identify key support/resistance levels.
# Enters long when price breaks above H3 with volume confirmation and 1w EMA50 uptrend.
# Enters short when price breaks below L3 with volume confirmation and 1w EMA50 downtrend.
# Weekly trend filter provides strong directional bias, reducing false breakouts in chop.
# Designed for 7-25 trades/year (~30-100 total over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via breakdowns with trend alignment.

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    camarilla_range = high_1w - low_1w
    h3_1w = close_1w + 1.1 * camarilla_range
    l3_1w = close_1w - 1.1 * camarilla_range
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1w bar)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Get 1w data for EMA50 trend filter - ONCE before loop
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 AND volume spike AND 1w EMA50 uptrend
            if (close[i] > h3_1w_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below L3 AND volume spike AND 1w EMA50 downtrend
            elif (close[i] < l3_1w_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla H3/L3 range OR trend reverses
            if (close[i] >= l3_1w_aligned[i] and close[i] <= h3_1w_aligned[i]) or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Camarilla H3/L3 range OR trend reverses
            if (close[i] >= l3_1w_aligned[i] and close[i] <= h3_1w_aligned[i]) or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals