#!/usr/bin/env python3
# 12h_1d_1w_Camarilla_R4S4_Breakout_Trend_Filter
# Hypothesis: Uses 1d and 1w Camarilla R4/S4 levels as key support/resistance on 12h timeframe.
# Enters long when price breaks above R4 with bullish 1d/1w trend and volume confirmation.
# Enters short when price breaks below S4 with bearish 1d/1w trend and volume confirmation.
# Uses 1d EMA50 and 1w EMA200 as trend filters to avoid counter-trend trades.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following multi-timeframe trend while using daily/weekly
# Camarilla breakouts for precise entries.

name = "12h_1d_1w_Camarilla_R4S4_Breakout_Trend_Filter"
timeframe = "12h"
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
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day (R4/S4)
    camarilla_r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each week (R4/S4)
    camarilla_r4_1w = close_1w + ((high_1w - low_1w) * 1.1 / 2)
    camarilla_s4_1w = close_1w - ((high_1w - low_1w) * 1.1 / 2)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly EMA200 for trend filter (longer-term trend)
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(camarilla_r4_1d_aligned[i]) or
            np.isnan(camarilla_s4_1d_aligned[i]) or
            np.isnan(camarilla_r4_1w_aligned[i]) or
            np.isnan(camarilla_s4_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Multi-timeframe trend filter: bullish if price > both EMAs, bearish if price < both
        trend_bullish = close[i] > ema_50_1d_aligned[i] and close[i] > ema_200_1w_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i] and close[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R4 (1d OR 1w) + bullish trend + volume spike
            if ((close[i] > camarilla_r4_1d_aligned[i] or close[i] > camarilla_r4_1w_aligned[i]) and
                trend_bullish and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 (1d OR 1w) + bearish trend + volume spike
            elif ((close[i] < camarilla_s4_1d_aligned[i] or close[i] < camarilla_s4_1w_aligned[i]) and
                  trend_bearish and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 (1d OR 1w) OR bearish trend
            if (close[i] < camarilla_s4_1d_aligned[i]) or \
               (close[i] < camarilla_s4_1w_aligned[i]) or \
               (not trend_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4 (1d OR 1w) OR bullish trend
            if (close[i] > camarilla_r4_1d_aligned[i]) or \
               (close[i] > camarilla_r4_1w_aligned[i]) or \
               (not trend_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals