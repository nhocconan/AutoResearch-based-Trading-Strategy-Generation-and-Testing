#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v2
# Uses daily Camarilla pivot levels (H3/L3) as breakout triggers on 4h chart.
# Long when price closes above H3 with volume > 1.3x 20-period average.
# Short when price closes below L3 with volume > 1.3x 20-period average.
# Exits when price returns to daily pivot point (PP) or touches opposite H3/L3 level.
# Uses 1d trend filter: only take longs when 4h close > 200-period EMA on 1d,
# only take shorts when 4h close < 200-period EMA on 1d.
# Designed for low trade frequency (<50/year) with strong trend alignment.
# Works in bull markets via breakouts with trend, in bear via mean reversion to pivot.

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily OHLC for previous day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels (H3/L3 used for breakouts)
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla H3 = PP + 1.1/4 * range, L3 = PP - 1.1/4 * range
    h3 = pp + (1.1 / 4) * range_1d
    l3 = pp - (1.1 / 4) * range_1d
    
    # Align daily levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Daily 200 EMA for trend filter (only use longs in uptrend, shorts in downtrend)
    close_1d_series = pd.Series(df_1d['close'])
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.3 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price closes above H3 AND in uptrend (close > daily EMA200)
        if close[i] > h3_aligned[i] and close[i] > ema_200_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short conditions: price closes below L3 AND in downtrend (close < daily EMA200)
        elif close[i] < l3_aligned[i] and close[i] < ema_200_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to daily pivot (PP) or touches opposite level
        elif position == 1 and (close[i] <= pp_aligned[i] or close[i] <= l3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= pp_aligned[i] or close[i] >= h3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals