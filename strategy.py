#!/usr/bin/env python3
# 4h_1d_VWAP_Pullback_Trend_Follow
# Hypothesis: On 4h timeframe, buy pullbacks to VWAP during daily uptrend, sell rallies to VWAP during daily downtrend.
# Uses 1d VWAP and 1d EMA50 for trend filter. VWAP acts as dynamic support/resistance.
# Trend filter ensures we only trade with the daily trend, reducing whipsaw in ranging markets.
# Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag.
# Works in bull/bear via daily trend filter - only trade with the daily trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_VWAP_Pullback_Trend_Follow"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for VWAP and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily VWAP ===
    # VWAP = sum(price * volume) / sum(volume) for the day
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vp_1d = typical_price_1d * df_1d['volume']
    cum_vp_1d = vp_1d.cumsum()
    cum_vol_1d = df_1d['volume'].cumsum()
    vwap_1d = cum_vp_1d / cum_vol_1d
    
    # === Calculate daily EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Distance to VWAP as percentage ===
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3.0
    # We'll use price action relative to VWAP, no need to calculate 4h VWAP
    
    # Align all daily levels to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d.values)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        vwap_1d_val = vwap_1d_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vwap_1d_val) or np.isnan(ema50_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price pulls back to VWAP during daily uptrend
            if (close_val <= vwap_1d_val * 1.005 and  # Near or slightly above VWAP (allow 0.5% buffer)
                close_val >= vwap_1d_val * 0.995 and  # Near or slightly below VWAP
                close_val > ema50_1d_val):  # Only in daily uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price rallies to VWAP during daily downtrend
            elif (close_val >= vwap_1d_val * 0.995 and  # Near or slightly below VWAP
                  close_val <= vwap_1d_val * 1.005 and  # Near or slightly above VWAP
                  close_val < ema50_1d_val):  # Only in daily downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price moves significantly above VWAP or trend changes
            if close_val >= vwap_1d_val * 1.02:  # 2% above VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price moves significantly below VWAP or trend changes
            if close_val <= vwap_1d_val * 0.98:  # 2% below VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals