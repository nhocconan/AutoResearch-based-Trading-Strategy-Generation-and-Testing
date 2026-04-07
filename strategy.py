#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Camarilla pivot levels on 1d provide high-probability reversal points in both bull and bear markets, 
while 1w trend filter ensures we trade with the higher timeframe momentum. Volume confirmation filters false breaks. 
Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    # Previous day's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First day: use current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels for previous day
    range_val = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_val / 2
    camarilla_h3 = prev_close + 1.1 * range_val / 4
    camarilla_h2 = prev_close + 1.1 * range_val / 6
    camarilla_h1 = prev_close + 1.1 * range_val / 12
    camarilla_l1 = prev_close - 1.1 * range_val / 12
    camarilla_l2 = prev_close - 1.1 * range_val / 6
    camarilla_l3 = prev_close - 1.1 * range_val / 4
    camarilla_l4 = prev_close - 1.1 * range_val / 2
    
    # Weekly EMA for trend filter (20-period)
    ema_20 = df_1w['close'].ewm(span=20, adjust=False).mean()
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20.values)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price below camarilla_l3 or trend turns bearish
            if close[i] < camarilla_l3[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price above camarilla_h3 or trend turns bullish
            if close[i] > camarilla_h3[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches camarilla_l4 with volume and bullish weekly trend
            if (close[i] <= camarilla_l4[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches camarilla_h4 with volume and bearish weekly trend
            elif (close[i] >= camarilla_h4[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals