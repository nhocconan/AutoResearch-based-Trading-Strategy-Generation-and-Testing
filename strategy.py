#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Previous day's pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Pivot support/resistance levels
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 6)
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 1d EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 4h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trend filter: price above/below 1d EMA34
            uptrend = close[i] > ema34_1d_aligned[i]
            downtrend = close[i] < ema34_1d_aligned[i]
            
            if uptrend:
                # Uptrend: breakout above R1
                long_cond = (close[i] > r1_4h[i] and volume[i] > vol_ma20[i])
                if long_cond:
                    signals[i] = 0.30
                    position = 1
            elif downtrend:
                # Downtrend: breakdown below S1
                short_cond = (close[i] < s1_4h[i] and volume[i] > vol_ma20[i])
                if short_cond:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long exit: trend reversal or breakdown below S1
            if close[i] < ema34_1d_aligned[i] or close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: trend reversal or breakout above R1
            if close[i] > ema34_1d_aligned[i] or close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla pivot breakout strategy on 4h timeframe.
# Uses 1d EMA34 for trend filter and 1d Camarilla R1/S1 levels for entries.
# In uptrends (price > 1d EMA34): buy breakouts above R1 with volume confirmation.
# In downtrends (price < 1d EMA34): sell breakdowns below S1 with volume confirmation.
# Exits on trend reversal (price crosses 1d EMA34) or price returns to S1/R1.
# Designed to work in both bull (trend following breaks) and bear (trend following breaks down).
# Targets 75-200 trades over 4 years to minimize fee drag. Uses discrete sizing (0.30).