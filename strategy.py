#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike Confirmation
Hypothesis: Camarilla pivot levels (R3/S3) act as strong intraday support/resistance. A breakout above R3 or below S3 with 1d EMA34 trend alignment and volume spike (>1.5x 20-bar vol MA) captures strong momentum with controlled trade frequency. Works in bull markets via upside breakouts and in bear markets via downside breakdowns. Discrete sizing (0.25) limits fee drag. Target: 75-200 total trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need 34 for EMA + 1 for safety
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(35, 20)  # 35 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla pivot levels for today (using previous day's OHLC)
        # Camarilla levels: based on previous day's range
        if i >= 1:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_range = prev_high - prev_low
            
            # Camarilla levels
            R3 = prev_close + (prev_range * 1.1 / 4)
            S3 = prev_close - (prev_range * 1.1 / 4)
        else:
            # Not enough data for previous day
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above R3 + price above 1d EMA34 + volume confirmation
            long_signal = (curr_close > R3) and (curr_close > ema_34_val) and volume_confirm
            # Short: break below S3 + price below 1d EMA34 + volume confirmation
            short_signal = (curr_close < S3) and (curr_close < ema_34_val) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below R3 OR price crosses below 1d EMA34
            if (curr_close < R3) or (curr_close < ema_34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above S3 OR price crosses above 1d EMA34
            if (curr_close > S3) or (curr_close > ema_34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0