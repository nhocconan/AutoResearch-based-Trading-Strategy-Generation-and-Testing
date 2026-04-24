#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1w pivot trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation and Camarilla levels, 1w for pivot trend direction.
- Camarilla levels: R3/S3 as breakout continuation zones (price breaking R3/S3 with volume continues trend).
- Entry: Long when price breaks above R3 AND volume > 2.0 * 20-period average volume AND 1w pivot trend is up.
         Short when price breaks below S3 AND volume > 2.0 * 20-period average volume AND 1w pivot trend is down.
- Exit: Opposite Camarilla breakout (R4/S4) or time-based (max 10 bars hold).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via breakout continuation and bear markets via short breakdowns.
- Volume confirmation avoids false breakouts. Weekly pivot trend ensures trading with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    typical = (high + low + close) / 3.0
    range_val = high - low
    R4 = close + range_val * 1.1 / 2.0
    R3 = close + range_val * 1.1 / 4.0
    S3 = close - range_val * 1.1 / 4.0
    S4 = close - range_val * 1.1 / 2.0
    return R4, R3, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA and Camarilla
        return np.zeros(n)
    
    r4_1d, r3_1d, s3_1d, s4_1d = camarilla(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d volume average for confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1w pivot trend (Higher High/Higher Low for uptrend, Lower High/Lower Low for downtrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weeks for trend
        return np.zeros(n)
    
    # Simple trend: compare current week close to previous week close
    close_1w = df_1w['close'].values
    trend_1w = np.zeros(len(close_1w))
    trend_1w[0] = 0  # No trend for first week
    for i in range(1, len(close_1w)):
        if close_1w[i] > close_1w[i-1]:
            trend_1w[i] = 1   # Uptrend
        elif close_1w[i] < close_1w[i-1]:
            trend_1w[i] = -1  # Downtrend
        else:
            trend_1w[i] = trend_1w[i-1]  # Same as previous
    
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0  # For time-based exit
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(trend_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Increment bars in trade
        if position != 0:
            bars_in_trade += 1
        
        # Exit conditions: opposite Camarilla breakout or max hold time (10 bars)
        if position != 0:
            exit_signal = False
            # Exit long: price breaks below S3
            if position == 1:
                if curr_low <= s3_1d_aligned[i]:
                    exit_signal = True
            # Exit short: price breaks above R3
            elif position == -1:
                if curr_high >= r3_1d_aligned[i]:
                    exit_signal = True
            
            # Time-based exit: max 10 bars (~60 hours)
            if bars_in_trade >= 10:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
                continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and 1w pivot trend filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= r3_1d_aligned[i] and prev_close < r3_1d_aligned[i-1]
            breakout_down = curr_low <= s3_1d_aligned[i] and prev_close > s3_1d_aligned[i-1]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # 1w pivot trend filter: only trade in direction of weekly trend
            trend_filter_long = trend_1w_aligned[i] >= 0   # Allow uptrend and sideways
            trend_filter_short = trend_1w_aligned[i] <= 0  # Allow downtrend and sideways
            
            if breakout_up and volume_confirm and trend_filter_long:
                signals[i] = 0.25
                position = 1
                bars_in_trade = 0
            elif breakout_down and volume_confirm and trend_filter_short:
                signals[i] = -0.25
                position = -1
                bars_in_trade = 0
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wPivotTrend_v1"
timeframe = "6h"
leverage = 1.0