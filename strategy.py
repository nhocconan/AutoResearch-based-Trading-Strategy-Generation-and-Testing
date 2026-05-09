#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly volume-weighted average price (VWAP) and daily trend filter.
# Uses weekly VWAP as dynamic support/resistance and daily EMA50 for trend filter.
# Weekly VWAP adapts to market conditions and works in both bull and bear markets.
# Daily trend filter reduces whipsaw by only allowing trades in direction of higher timeframe trend.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_VWAP_1w_VWAP_1dEMA50_Trend"
timeframe = "1d"
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
    
    # Calculate weekly VWAP from previous week
    # VWAP = sum(price * volume) / sum(volume) for the week
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Calculate cumulative sums for the week (7 days)
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            cum_tpv[i] = tpv[i]
            cum_vol[i] = volume[i]
        else:
            cum_tpv[i] = cum_tpv[i-1] + tpv[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
    
    # Get values from 7 days ago (previous week)
    prev_cum_tpv = np.roll(cum_tpv, 7)
    prev_cum_vol = np.roll(cum_vol, 7)
    prev_cum_tpv[:7] = np.nan
    prev_cum_vol[:7] = np.nan
    
    weekly_vwap = prev_cum_tpv / prev_cum_vol
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_vwap[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly VWAP + daily uptrend + volume spike
            if close[i] > weekly_vwap[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly VWAP + daily downtrend + volume spike
            elif close[i] < weekly_vwap[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly VWAP or trend reversal
            if close[i] <= weekly_vwap[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly VWAP or trend reversal
            if close[i] >= weekly_vwap[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals