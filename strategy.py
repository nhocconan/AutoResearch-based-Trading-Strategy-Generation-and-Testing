#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: Daily Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide high-probability reversal zones. 
# Weekly EMA filter ensures trades align with higher timeframe trend. Volume confirmation 
# filters false breaks. Designed for low trade frequency (~10-25/year) to minimize fee drag.
# Works in bull markets via buying dips in uptrend and selling rallies in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Camarilla pivot levels (based on previous day)
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Shift to get previous day's values
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_typical = (prev_high + prev_low + prev_close) / 3.0
    # First day: use current values
    prev_typical[0] = typical_price[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels
    range_val = prev_high - prev_low
    # Resistance levels
    r1 = prev_close + range_val * 1.1 / 12
    r2 = prev_close + range_val * 1.1 / 6
    r3 = prev_close + range_val * 1.1 / 4
    r4 = prev_close + range_val * 1.1 / 2
    # Support levels
    s1 = prev_close - range_val * 1.1 / 12
    s2 = prev_close - range_val * 1.1 / 6
    s3 = prev_close - range_val * 1.1 / 4
    s4 = prev_close - range_val * 1.1 / 2
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(prev_typical[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Trend filter: price above/below weekly EMA21
        trend_bullish = close[i] > ema_21_1w_aligned[i]
        trend_bearish = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions
        # Long: Price touches S3/S4 support in uptrend with volume confirmation
        if trend_bullish and vol_confirm and position != 1:
            if low[i] <= s3[i] or low[i] <= s4[i]:
                position = 1
                signals[i] = 0.25
        # Short: Price touches R3/R4 resistance in downtrend with volume confirmation
        elif trend_bearish and vol_confirm and position != -1:
            if high[i] >= r3[i] or high[i] >= r4[i]:
                position = -1
                signals[i] = -0.25
        # Exit: Opposite touch or trend reversal
        elif position == 1 and (high[i] >= r1[i] or not trend_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= s1[i] or not trend_bearish):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals