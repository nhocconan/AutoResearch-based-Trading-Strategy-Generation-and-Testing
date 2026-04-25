#!/usr/bin/env python3
"""
12h Camarilla R3S3 Breakout + 1w EMA50 Trend + Volume Spike + Chop Regime Filter
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance on 12h timeframe. 
Breakouts above R3 or below S3 with 1w EMA50 trend filter capture momentum moves. 
Volume confirmation ensures institutional participation. Chop filter (BBW percentile) avoids 
whipsaws in low-volatility ranging markets. Discrete sizing (0.25) controls drawdown. 
Designed for BTC/ETH in both bull/bear regimes via trend filter and high-probability entries.
Target: 50-150 trades over 4 years (12-37/year) on 12h timeframe.
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load 1d data ONCE before loop for Camarilla pivot levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d Camarilla levels (R3, S3) - based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d Bollinger Bands for chop regime (20, 2)
    bb_ma = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_percentile = bb_width / (bb_width_ma + 1e-10)
    chop_filter = bb_width_percentile > 0.5  # Avoid low volatility squeeze
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(chop_filter_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend + volume + chop filter
            # Long: price breaks above Camarilla R3 AND bullish bias AND volume spike AND chop filter
            long_entry = (curr_high > camarilla_r3_aligned[i]) and bullish_bias and vol_spike and chop_filter_aligned[i]
            # Short: price breaks below Camarilla S3 AND bearish bias AND volume spike AND chop filter
            short_entry = (curr_low < camarilla_s3_aligned[i]) and bearish_bias and vol_spike and chop_filter_aligned[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla S3 (reversal) OR loss of bullish bias
            if (curr_low < camarilla_s3_aligned[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla R3 (reversal) OR loss of bearish bias
            if (curr_high > camarilla_r3_aligned[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0