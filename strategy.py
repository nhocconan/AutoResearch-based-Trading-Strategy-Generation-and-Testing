#!/usr/bin/env python3
"""
Hypothesis: 4-hour Volume-Weighted Average Price (VWAP) deviation with 1-day trend filter and volume confirmation.
Price deviations from VWAP indicate mean-reversion opportunities in ranging markets, while strong deviations
with volume confirm institutional activity. The 1-day trend filter ensures trades align with the daily trend.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

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
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    
    # Calculate 20-period VWAP
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    tp_vol_sum = pd.Series(tp_vol).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(vol_sum != 0, tp_vol_sum / vol_sum, np.nan)
    
    # Calculate standard deviation of TP from VWAP for volatility bands
    tp_dev = typical_price - vwap
    # Use 20-period rolling std dev of TP deviation
    tp_dev_ma = pd.Series(tp_dev).rolling(window=20, min_periods=20).mean().values
    tp_dev_sq = tp_dev * tp_dev
    tp_dev_var = pd.Series(tp_dev_sq).rolling(window=20, min_periods=20).mean().values - (tp_dev_ma * tp_dev_ma)
    tp_dev_std = np.sqrt(np.maximum(tp_dev_var, 0))
    
    # Upper and lower bands (2 standard deviations)
    upper_band = vwap + 2.0 * tp_dev_std
    lower_band = vwap - 2.0 * tp_dev_std
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower band with volume, in uptrend
            if (close[i] <= lower_band[i] and                    # Price at or below lower band
                close[i] > ema_50_1d_aligned[i] and              # Above 1d EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):                # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: price touches upper band with volume, in downtrend
            elif (close[i] >= upper_band[i] and                  # Price at or above upper band
                  close[i] < ema_50_1d_aligned[i] and            # Below 1d EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):              # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP or crosses 1d EMA in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above VWAP or below 1d EMA
                if close[i] >= vwap[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below VWAP or above 1d EMA
                if close[i] <= vwap[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_Deviation_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0