#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolConfirm_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume confirmation.
Use 4h for signal direction (trend filter), 1d for volume regime filter, and 1h only for entry timing precision.
Camarilla R1/S1 provide intraday support/resistance levels derived from prior 1h candle.
In bull markets (price > 4h EMA50): long when price breaks above R1 and 1d volume > 1.5x 20-period average.
In bear markets (price < 4h EMA50): short when price breaks below S1 and 1d volume > 1.5x 20-period average.
Exit on opposite Camarilla level touch or trend reversal.
Position size: 0.20 to minimize fee churn and manage drawdown.
Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period average 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate prior 1h Camarilla levels (using prior 1h bar's high/low/close)
    # Shift high/low/close by 1 to avoid look-ahead (use prior completed 1h bar)
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    hl_range_prior = high_shift - low_shift
    # 1h Camarilla R1 and S1 (key intraday resistance/support)
    r1_1h = close_shift + (1.1 * hl_range_prior / 12)  # R1 = prior_close + 1.1*(prior_high-prior_low)/12
    s1_1h = close_shift - (1.1 * hl_range_prior / 12)  # S1 = prior_close - 1.1*(prior_high-prior_low)/12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(r1_1h[i]) or
            np.isnan(s1_1h[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above 4h EMA50)
        htf_4h_bullish = close[i] > ema_50_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # Use aligned 1d volume MA for current 1h bar
        volume_confirm = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(volume_1d_aligned[i]) else False
        
        if position == 0:
            # Long setup: price breaks above prior 1h Camarilla R1 + 4h uptrend + 1d volume confirmation
            long_setup = (close[i] > r1_1h[i]) and htf_4h_bullish and volume_confirm
            
            # Short setup: price breaks below prior 1h Camarilla S1 + 4h downtrend + 1d volume confirmation
            short_setup = (close[i] < s1_1h[i]) and htf_4h_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches prior 1h Camarilla S1 (stop) OR 4h trend turns bearish
            if (close[i] <= s1_1h[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches prior 1h Camarilla R1 (stop) OR 4h trend turns bullish
            if (close[i] >= r1_1h[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolConfirm_v1"
timeframe = "1h"
leverage = 1.0