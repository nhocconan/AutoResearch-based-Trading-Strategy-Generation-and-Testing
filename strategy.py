#!/usr/bin/env python3
# 1d_Camarilla_Pivot_H4Trend_Volume
# Hypothesis: On daily timeframe, price reacting to Camarilla pivot levels (H3/L3) with weekly trend filter and volume confirmation works in both bull and bear markets.
# Weekly trend avoids counter-trend trades, volume reduces false signals. Camarilla levels provide precise support/resistance.
# Designed for low frequency (~10-25 trades/year) to minimize fee drag.

name = "1d_Camarilla_Pivot_H4Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    # Using previous day's values to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for today using yesterday's HLC
    # H4 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 6
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    rang = prev_high - prev_low
    H4 = prev_close + 1.1 * rang / 2
    L3 = prev_close - 1.1 * rang / 6
    L4 = prev_close - 1.1 * rang / 2
    H3 = prev_close + 1.1 * rang / 4
    
    # Weekly trend: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation: 20-day average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H4[i]) or np.isnan(L3[i]) or np.isnan(H3[i]) or np.isnan(L4[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price at L3 or L4 with weekly uptrend and volume
            if ((close[i] <= L3[i] * 1.002 or close[i] <= L4[i] * 1.002) and 
                trend_1w_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price at H3 or H4 with weekly downtrend and volume
            elif ((close[i] >= H3[i] * 0.998 or close[i] >= H4[i] * 0.998) and 
                  trend_1w_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price reaches H3 or H4 or trend fails
            if (close[i] >= H3[i] * 0.998 or 
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price reaches L3 or L4 or trend fails
            if (close[i] <= L3[i] * 1.002 or 
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals