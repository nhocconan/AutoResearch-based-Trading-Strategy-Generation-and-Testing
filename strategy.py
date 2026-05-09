#!/usr/bin/env python3
# Hypothesis: 12h timeframe with daily volume-weighted average price (VWAP) and weekly trend filter.
# Uses daily VWAP for mean-reversion entries when price deviates significantly, filtered by weekly EMA trend.
# Weekly trend filter ensures trades align with higher timeframe direction, reducing whipsaw in sideways markets.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.

name = "12h_DailyVWAP_MeanReversion_1wTrend"
timeframe = "12h"
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
    
    # Calculate daily VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = typical_price * volume
    vwap_den = volume
    
    # Use expanding window for cumulative VWAP, reset daily
    # Since we don't have date grouping, use 2-period approximation for daily VWAP
    # More robust: calculate VWAP since start of day using intraday data approximation
    # For 12h timeframe, we approximate daily VWAP using 2-period cumulative sum
    vwap_cum_num = np.nancumsum(vwap_num)
    vwap_cum_den = np.nancumsum(vwap_den)
    vwap = vwap_cum_num / vwap_cum_den
    # Handle division by zero
    vwap = np.where(vwap_cum_den != 0, vwap, np.nan)
    
    # Deviation from VWAP as percentage
    vwap_dev = (close - vwap) / vwap
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_dev[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price significantly below VWAP (-1.5%) + weekly uptrend + volume confirmation
            if vwap_dev[i] < -0.015 and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price significantly above VWAP (+1.5%) + weekly downtrend + volume confirmation
            elif vwap_dev[i] > 0.015 and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or trend reversal
            if vwap_dev[i] >= -0.005 or not trend_up[i]:  # Close to VWAP or trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP or trend reversal
            if vwap_dev[i] <= 0.005 or not trend_down[i]:  # Close to VWAP or trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals