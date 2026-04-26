#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v6
Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter (EMA34) and volume spike confirmation.
Goes long when price breaks above R1 AND close > 1d EMA34 AND volume > 1.5x 20-bar average.
Goes short when price breaks below S1 AND close < 1d EMA34 AND volume > 1.5x 20-bar average.
Exits on opposite Camarilla level touch (R3/S3) or when price re-enters the R1-S1 range.
Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters counter-trend trades.
Volume spike ensures breakout has conviction. Designed for 4h timeframe targeting 20-50 trades/year.
Works in bull/bear markets by trading with 1d trend and using volume to avoid false breakouts.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # PP = (high+low+close)/3, S1 = close - 0.275*(high-low), etc.
    # We'll use the 1d data to compute levels for the 4h bars
    
    # For each 4h bar, we need the previous 1d bar's OHLC
    # Since we have aligned 1d data, we can use the previous completed 1d bar
    
    # Calculate typical price for 1d bars
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = close + 0.275*(high-low), S1 = close - 0.275*(high-low)
    # R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r1_1d = close_1d_arr + 0.275 * range_1d
    s1_1d = close_1d_arr - 0.275 * range_1d
    r3_1d = close_1d_arr + 1.1 * range_1d
    s3_1d = close_1d_arr - 1.1 * range_1d
    
    # Align Camarilla levels to 4h timeframe (use previous completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume spike: volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > r1_aligned[i]
        bearish_breakout = close[i] < s1_aligned[i]
        
        # Trend filter
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Exit conditions: touch opposite level or re-enter R1-S1 range
        touch_r3 = close[i] >= r3_aligned[i]
        touch_s3 = close[i] <= s3_aligned[i]
        in_range = (close[i] > s1_aligned[i]) and (close[i] < r1_aligned[i])
        
        if position == 0:
            # Long: bullish breakout AND uptrend AND volume spike
            if bullish_breakout and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND downtrend AND volume spike
            elif bearish_breakout and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: touch R3 OR bearish breakout OR re-enter range
            if touch_r3 or bearish_breakout or in_range:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: touch S3 OR bullish breakout OR re-enter range
            if touch_s3 or bullish_breakout or in_range:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v6"
timeframe = "4h"
leverage = 1.0