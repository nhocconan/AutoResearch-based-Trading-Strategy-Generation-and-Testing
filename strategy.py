#!/usr/bin/env python3

"""
Hypothesis: 6-hour VWAP Reversion with Weekly Trend Filter and Volume Confirmation.
Trades reversals at VWAP (volume-weighted average price) when price deviates significantly from VWAP,
filtered by weekly EMA trend direction and confirmed by volume spikes. Uses mean-reversion at VWAP
deviations as the edge, which works in both bull and bear markets by aligning with higher timeframe
trend. Designed for low trade frequency (15-35 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP for given arrays."""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    # Cumulative VWAP
    cum_vwap_num = np.cumsum(vwap_numerator)
    cum_vwap_den = np.cumsum(vwap_denominator)
    # Avoid division by zero
    vwap = np.divide(cum_vwap_num, cum_vwap_den, out=np.full_like(cum_vwap_num, np.nan), where=cum_vwap_den!=0)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (20-period)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate VWAP
    vwap = calculate_vwap(high, low, close, volume)
    
    # VWAP deviation bands (2 standard deviations of price-VWAP difference)
    price_vwap_diff = close - vwap
    # Use 50-period rolling std of the difference
    price_vwap_diff_series = pd.Series(price_vwap_diff)
    vwap_std = price_vwap_diff_series.rolling(window=50, min_periods=50).std().values
    vwap_upper = vwap + 2.0 * vwap_std
    vwap_lower = vwap - 2.0 * vwap_std
    
    # Volume spike: current volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vwap_upper[i]) or 
            np.isnan(vwap_lower[i]) or np.isnan(vwap_std[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_30[i]
        
        if position == 0 and vol_spike:
            # Long: price touches/vwap_lower and weekly trend is up
            if close[i] <= vwap_lower[i] and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches/vwap_upper and weekly trend is down
            elif close[i] >= vwap_upper[i] and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP or weekly trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above VWAP or weekly trend turns down
                if close[i] >= vwap[i] or close[i] < ema_20_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below VWAP or weekly trend turns up
                if close[i] <= vwap[i] or close[i] > ema_20_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_VWAP_Reversion_1wEMA20_Volume"
timeframe = "6h"
leverage = 1.0