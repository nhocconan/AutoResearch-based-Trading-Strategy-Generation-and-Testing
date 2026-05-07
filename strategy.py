#!/usr/bin/env python3
"""
1d_VWAP_Deviation_With_WeeklyTrend
Hypothesis: Trade intraday VWAP deviations on daily chart with weekly trend filter. 
In bull markets (price > weekly VWAP), go long when price deviates below daily VWAP with volume confirmation. 
In bear markets (price < weekly VWAP), go short when price deviates above daily VWAP with volume confirmation.
Weekly trend filter avoids counter-trend trades; VWAP deviation captures mean reversion within trend.
Designed for 10-25 trades/year to minimize fee drag. Works in bull/bear via weekly trend filter.
"""

name = "1d_VWAP_Deviation_With_WeeklyTrend"
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
    
    # Get weekly data for trend filter (VWAP)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly VWAP for trend filter
    typical_price_weekly = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_weekly = (typical_price_weekly * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_weekly_array = vwap_weekly.values
    vwap_weekly_aligned = align_htf_to_ltf(prices, df_1w, vwap_weekly_array, additional_delay_bars=0)
    
    # Calculate daily VWAP for signal generation
    typical_price_daily = (high + low + close) / 3.0
    vwap_daily = (typical_price_daily * volume).cumsum() / volume.cumsum()
    vwap_daily_array = vwap_daily.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_weekly_aligned[i]) or 
            np.isnan(vwap_daily_array[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        price_above_weekly_vwap = close[i] > vwap_weekly_aligned[i]
        price_below_weekly_vwap = close[i] < vwap_weekly_aligned[i]
        
        # Daily VWAP deviation
        vwap_deviation = (close[i] - vwap_daily_array[i]) / vwap_daily_array[i]
        
        if position == 0:
            # Long: price below daily VWAP in weekly uptrend with volume spike
            if price_above_weekly_vwap and vwap_deviation < -0.005 and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price above daily VWAP in weekly downtrend with volume spike
            elif price_below_weekly_vwap and vwap_deviation > 0.005 and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily VWAP or weekly trend changes
            if vwap_deviation > -0.002 or not price_above_weekly_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily VWAP or weekly trend changes
            if vwap_deviation < 0.002 or not price_below_weekly_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals