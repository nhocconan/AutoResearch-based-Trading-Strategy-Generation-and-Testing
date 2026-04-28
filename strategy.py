#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above Camarilla R3, 1d EMA34 trending up, and volume > 1.5x 20-bar average.
# Enter short when price breaks below Camarilla S3, 1d EMA34 trending down, and volume > 1.5x 20-bar average.
# Exit when price reaches the opposite Camarilla level (S3 for long, R3 for short) or crosses the 1d EMA34.
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 80-160 total trades over 4 years (20-40/year) to avoid excessive fee churn.
# Camarilla provides precise intraday support/resistance; EMA34 filters for 1d trend alignment; volume confirms breakout strength.

name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla levels are based on previous day's range
    # We need to calculate these for each 4h bar using the previous day's data
    # Since we're on 4h timeframe, we'll use the daily OHLC from 1d timeframe
    
    # Get previous day's close for Camarilla calculation
    # Camarilla levels: 
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.0*(high-low)
    # R2 = close + 0.75*(high-low)
    # R1 = close + 0.5*(high-low)
    # PP = (high+low+close)/3
    # S1 = close - 0.5*(high-low)
    # S2 = close - 0.75*(high-low)
    # S3 = close - 1.0*(high-low)
    # S4 = close - 1.5*(high-low)
    
    # For intraday trading, we use previous day's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    
    # Handle first bar
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    
    # Calculate Camarilla levels based on previous day's data
    prev_range = prev_high_1d - prev_low_1d
    camarilla_pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_r3 = prev_close_1d + 1.0 * prev_range
    camarilla_s3 = prev_close_1d - 1.0 * prev_range
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        # Exit conditions: price reaches opposite Camarilla level or crosses 1d EMA34
        exit_long = close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_aligned[i]
        exit_short = close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_aligned[i]
        
        # Handle entries and exits
        if breakout_up and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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