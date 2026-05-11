#!/usr/bin/env python3
"""
6h_1w_WeeklyTrend_DailyMeanReversion
Hypothesis: Uses weekly trend direction (price above/below weekly EMA200) and daily mean reversion (price deviation from daily VWAP) to enter trades on 6h timeframe. In weekly uptrend, go long when price deviates significantly below daily VWAP; in weekly downtrend, go short when price deviates significantly above daily VWAP. Exits when price returns to VWAP or weekly trend reverses. Designed to work in both bull and bear markets by aligning with weekly trend while exploiting daily mean reversion. Targets 15-30 trades/year via strict entry conditions requiring both trend alignment and significant deviation.
"""

name = "6h_1w_WeeklyTrend_DailyMeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Trend Filter (EMA200) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(
        span=200, adjust=False, min_periods=200
    ).mean().values
    weekly_trend_up = align_htf_to_ltf(prices, df_1w, ema_200_1w)  # price > ema200 = uptrend
    
    # --- Daily VWAP for Mean Reversion ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP: cumulative (price * volume) / cumulative volume
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_1d = np.cumsum(typical_price_1d * df_1d['volume'].values) / np.cumsum(df_1d['volume'].values)
    # Handle division by zero on first bar
    vwap_1d = np.where(np.cumsum(df_1d['volume'].values) > 0, vwap_1d, typical_price_1d)
    vwap_6h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # --- Daily ATR for Deviation Threshold ---
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_trend_up[i]) or np.isnan(vwap_6h[i]) or 
            np.isnan(atr_6h[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Calculate deviation from VWAP in ATR units
        deviation = (close[i] - vwap_6h[i]) / atr_6h[i]
        
        if position == 0:
            # Long: weekly uptrend AND price significantly below VWAP (mean reversion long)
            if (close[i] > weekly_trend_up[i] and  # weekly uptrend
                deviation < -1.5):  # significantly below VWAP
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend AND price significantly above VWAP (mean reversion short)
            elif (close[i] < weekly_trend_up[i] and  # weekly downtrend
                  deviation > 1.5):  # significantly above VWAP
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to VWAP OR weekly trend turns down
                if (deviation >= -0.5) or (close[i] <= weekly_trend_up[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to VWAP OR weekly trend turns up
                if (deviation <= 0.5) or (close[i] >= weekly_trend_up[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals