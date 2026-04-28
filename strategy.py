#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d EMA34 trend filter with Camarilla R3/S3 breakout and volume confirmation.
# Enter long when price breaks above Camarilla R3 with volume > 2.0x 20-bar average and price > 1d EMA34 (uptrend).
# Enter short when price breaks below Camarilla S3 with volume > 2.0x 20-bar average and price < 1d EMA34 (downtrend).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 20-50 trades per year.
# Camarilla levels provide high-probability reversal points, volume confirms breakout strength, 1d EMA34 filters counter-trend noise.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R3, S3)
    def camarilla_levels(high, low, close):
        # Typical price for pivot
        pp = (high + low + close) / 3.0
        range_ = high - low
        r3 = pp + range_ * 1.1 / 4
        s3 = pp - range_ * 1.1 / 4
        return r3, s3
    
    # Calculate Camarilla levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        r3_1d[i], s3_1d[i] = camarilla_levels(high_1d[i], low_1d[i], close_1d[i])
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 4h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > r3_1d_aligned[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]
        short_breakout = close[i] < s3_1d_aligned[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < s3_1d_aligned[i]
        short_exit = close[i] > r3_1d_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals