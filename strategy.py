#!/usr/bin/env python3
# 12h_1d_Camarilla_Pivot_Support_Resistance_With_Volume_Confirmation
# Hypothesis: Trade reversals at 1d Camarilla R4/S4 levels with 1d trend filter and volume confirmation.
# In uptrends (1d close > EMA34), buy at S4 support; in downtrends (1d close < EMA34), sell at R4 resistance.
# Uses volume spike (volume > 1.5x 20-period average) to confirm institutional interest at pivot levels.
# Designed for low-frequency, high-conviction trades to minimize fee drag and work in both bull/bear markets.
# Targets ~20-30 trades/year on 12h timeframe.

name = "12h_1d_Camarilla_Pivot_Support_Resistance_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    # But standard Camarilla uses: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    # We'll use the previous day's range to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R4 and S4 from previous day
    camarilla_r4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_s4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for EMA and volume calculations
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at or below S4 support in uptrend with volume spike
            if (low[i] <= s4_aligned[i] * 1.002 and  # within 0.2% of S4
                trend_1d_up_aligned[i] > 0.5 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price at or above R4 resistance in downtrend with volume spike
            elif (high[i] >= r4_aligned[i] * 0.998 and  # within 0.2% of R4
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses above EMA34 (trend change) or RSI-like mean reversion
            if (close[i] > ema34_1d[i] * 1.01 or  # 1% above EMA suggests overextension
                high[i] >= r4_aligned[i] * 0.995):  # Touched R4 (opposite level)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses below EMA34 or touches S4
            if (close[i] < ema34_1d[i] * 0.99 or  # 1% below EMA
                low[i] <= s4_aligned[i] * 1.005):   # Touched S4 (opposite level)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals