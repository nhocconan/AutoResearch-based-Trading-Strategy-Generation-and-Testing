#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h with 12h EMA34 trend filter and volume spike confirmation. 
R3/S3 levels offer strong intraday support/resistance with fewer false breakouts than R4/S4. 
In bull markets: buy when price breaks above R3 and price > 12h EMA34. 
In bear markets: sell when price breaks below S3 and price < 12h EMA34. 
Requires volume > 2.0x 20-period average for confirmation to filter noise. 
Exit on opposite Camarilla level touch (R3/S3) or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 75-200 total trades over 4 years = 19-50/year. 
Weekly levels reduce noise and false breakouts, improving win rate in both bull and bear markets.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period average volume for confirmation (using 1d volume)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    # Daily Camarilla R3 and S3 levels
    r3_1d = close_1d + (1.1 * hl_range_1d / 6)  # R3 = close + 1.1*(high-low)/6
    s3_1d = close_1d - (1.1 * hl_range_1d / 6)  # S3 = close - 1.1*(high-low)/6
    
    # Align daily Camarilla levels to 4h prices
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA34)
        htf_12h_bullish = close[i] > ema_34_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_34_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above daily Camarilla R3 + 12h uptrend + volume confirmation
            long_setup = (close[i] > r3_aligned[i]) and htf_12h_bullish and volume_confirm
            
            # Short setup: price breaks below daily Camarilla S3 + 12h downtrend + volume confirmation
            short_setup = (close[i] < s3_aligned[i]) and htf_12h_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches daily Camarilla S3 (stop) OR 12h trend turns bearish
            if (close[i] <= s3_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches daily Camarilla R3 (stop) OR 12h trend turns bullish
            if (close[i] >= r3_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0