#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_1wTrend_Volume
Hypothesis: Buy breakouts above weekly Camarilla R4 level with volume confirmation when price is above 100-period EMA on daily (long-term uptrend). Sell breakdowns below weekly S4 level with volume confirmation when price is below 100-period EMA on daily (long-term downtrend). Uses weekly pivots for structure, daily EMA for trend filter, and volume for confirmation. Designed for very low frequency (<10 trades/year) to minimize fee drag on 1d timeframe. Works in bull by buying strong breakouts in uptrend, works in bear by selling breakdowns in downtrend.
"""

name = "1d_Camarilla_Pivot_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # === Get Weekly OHLC for Camarilla Calculation ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    prior_close = df_1w['close'].shift(1).values
    
    # Align to daily timeframe
    prior_high_1d = align_htf_to_ltf(prices, df_1w, prior_high)
    prior_low_1d = align_htf_to_ltf(prices, df_1w, prior_low)
    prior_close_1d = align_htf_to_ltf(prices, df_1w, prior_close)
    
    # Camarilla R4 and S4 levels (outer bands)
    R4 = prior_close_1d + (prior_high_1d - prior_low_1d) * 1.1 / 2
    S4 = prior_close_1d - (prior_high_1d - prior_low_1d) * 1.1 / 2
    
    # === Daily EMA100 Trend Filter ===
    ema100 = pd.Series(close).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # === Volume Spike Filter (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA and volume calculation)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R4[i]) or np.isnan(S4[i]) or 
            np.isnan(ema100[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above R4 with uptrend (close > EMA100) and volume spike
            if (close[i] > R4[i] and 
                close[i] > ema100[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Close crosses below S4 with downtrend (close < EMA100) and volume spike
            elif (close[i] < S4[i] and 
                  close[i] < ema100[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Close crosses back through the opposite Camarilla level
            if position == 1:
                if close[i] < S4[i]:  # Exit long if price breaks below S4
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > R4[i]:  # Exit short if price breaks above R4
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals