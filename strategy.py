#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Camarilla pivot levels (R3/S3) with 1w EMA50 trend filter and volume spike confirmation.
# Enter long when price breaks above Camarilla R3 with volume > 2.0x 20-bar average and price > 1w EMA50 (uptrend).
# Enter short when price breaks below Camarilla S3 with volume > 2.0x 20-bar average and price < 1w EMA50 (downtrend).
# Exit on close below Camarilla S3 (for longs) or above Camarilla R3 (for shorts).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 30-100 trades over 4 years.
# Camarilla levels provide precise intraday support/resistance, volume confirms breakout strength, 1w EMA50 filters counter-trend noise.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    def camarilla_levels(high, low, close):
        # Camarilla levels calculated from previous day's range
        range_ = high - low
        camarilla_r3 = close + range_ * 1.1 / 4
        camarilla_s3 = close - range_ * 1.1 / 4
        return camarilla_r3, camarilla_s3
    
    camarilla_r3 = np.full_like(close, np.nan)
    camarilla_s3 = np.full_like(close, np.nan)
    
    # Calculate Camarilla levels for each bar using previous day's OHLC
    for i in range(1, n):
        camarilla_r3[i], camarilla_s3[i] = camarilla_levels(high[i-1], low[i-1], close[i-1])
    
    # Calculate 1d volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > camarilla_r3[i] and volume_confirm[i] and close[i] > ema_50_1w_aligned[i]
        short_breakout = close[i] < camarilla_s3[i] and volume_confirm[i] and close[i] < ema_50_1w_aligned[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_s3[i]
        short_exit = close[i] > camarilla_r3[i]
        
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