# 4h_1d_Camarilla_R1_S1_Breakout_1dEMA50_Trend_Volume_v4
# Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries on 4h timeframe.
# Trend filtered by daily EMA50 to ensure alignment with higher timeframe direction.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Added Bollinger Band width filter to avoid choppy markets and reduce false breakouts.
# This version tightens entry conditions to reduce trade frequency and improve generalization.
# Works in bull/bear markets by following daily trend direction while using Camarilla levels for precise entries.
# Designed for low trade frequency (<200 total 4h trades) to minimize fee drag.

name = "4h_1d_Camarilla_R1_S1_Breakout_1dEMA50_Trend_Volume_v4"
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
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Bollinger Band width filter to avoid choppy markets (20-period, 2 std dev)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    # Only trade when BB width is above average (avoid low volatility chop)
    vol_filter = bb_width > bb_width_ma
    
    # Daily data for Camarilla pivot levels and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # Handle first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla formulas
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + (range_1d * 1.1 / 12)
    camarilla_s1 = prev_close - (range_1d * 1.1 / 12)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(bb_width[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 + volume spike + price above daily EMA50 + volatility filter
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i] and
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 + volume spike + price below daily EMA50 + volatility filter
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla R1 OR closes below daily EMA50
            if (close[i] < camarilla_r1_aligned[i]) or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla S1 OR closes above daily EMA50
            if (close[i] > camarilla_s1_aligned[i]) or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals