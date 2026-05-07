#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Filter_Volume
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 in 1d uptrend with volume spike, short when breaks below S3 in 1d downtrend with volume spike.
Exit on opposite breakout or trend reversal. Targets 12-37 trades/year on 12h timeframe.
Works in bull via breakouts, in bear via short breakdowns with trend filter reducing whipsaw.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Filter_Volume"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d close aligned for trend determination
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels on 12h using previous period's OHLC
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # We use previous bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_width = 1.1 * (prev_high - prev_low)
    r3 = prev_close + camarilla_width
    s3 = prev_close - camarilla_width
    
    # Volume confirmation: 20-period average on 12h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend determination
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 in 1d uptrend with volume confirmation
            if (close[i] > r3[i] and 
                trend_1d_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 in 1d downtrend with volume confirmation
            elif (close[i] < s3[i] and 
                  trend_1d_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or 1d trend turns down
            if (close[i] < s3[i] or not trend_1d_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R3 or 1d trend turns up
            if (close[i] > r3[i] or not trend_1d_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals