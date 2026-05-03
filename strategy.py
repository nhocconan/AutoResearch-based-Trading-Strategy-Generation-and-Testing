#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d volume spike and 12h trend filter.
# Bollinger Band squeeze (low volatility) precedes breakouts. Enter on breakout above/below
# BB with volume confirmation. Filter by 12h EMA50 trend direction and 1d volume > 2x 20-period MA.
# Works in bull via upside breakouts and bear via downside breakouts when aligned with 12h trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "6h_BB_Squeeze_Breakout_12hTrend_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width (for squeeze detection)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < (0.5 * bb_width_ma_50)  # BB width < 50% of 50-period average
    
    # Breakout conditions
    breakout_up = close > bb_upper
    breakout_down = close < bb_lower
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d volume spike: current volume > 2x 20-period MA
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > (2.0 * vol_ma_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        is_squeeze = squeeze_condition[i]
        is_breakout_up = breakout_up[i]
        is_breakout_down = breakout_down[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema50_12h_aligned[i]
        trend_down = close_val < ema50_12h_aligned[i]
        
        # Entry logic
        if position == 0:
            # Long: BB squeeze breakout up + volume spike + 12h uptrend
            if is_squeeze and is_breakout_up and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout down + volume spike + 12h downtrend
            elif is_squeeze and is_breakout_down and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: BB squeeze breakout down OR loss of 12h uptrend OR volume drops
            if is_breakout_down or not trend_up or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: BB squeeze breakout up OR loss of 12h downtrend OR volume drops
            if is_breakout_up or not trend_down or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals