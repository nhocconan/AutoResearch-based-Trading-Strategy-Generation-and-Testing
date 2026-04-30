#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 + 1d EMA34 uptrend + volume > 2.0x 20-bar average.
# Short when price breaks below S3 + 1d EMA34 downtrend + volume > 2.0x 20-bar average.
# Uses discrete position sizing (0.25) to minimize fee churn. Targets 50-150 total trades over 4 years.
# Camarilla levels from 1d provide institutional support/resistance; breakouts with volume confirm institutional participation.
# Works in both bull/bear: trend filter ensures we only trade with the 1d trend, avoiding counter-trend whipsaws.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (avoid look-ahead)
    # Camarilla: based on previous day's high, low, close
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 + uptrend + volume confirmation
            if curr_high > r3_aligned[i] and close[i] > ema_34_aligned[i] and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif curr_low < s3_aligned[i] and close[i] < ema_34_aligned[i] and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price closes below 1d EMA34 (trend change)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above 1d EMA34 (trend change)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals