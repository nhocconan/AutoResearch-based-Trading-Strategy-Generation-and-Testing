#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# Uses 4h EMA50 for trend direction (price > EMA50 = long bias, < EMA50 = short bias)
# Enters long when price breaks above Camarilla R3 with volume > 1.5x 20-period average
# Enters short when price breaks below Camarilla S3 with volume > 1.5x 20-period average
# Uses discrete position sizing 0.20 to minimize fee churn
# Targets 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with 4h trend direction
# Uses session filter (08-20 UTC) to avoid low-liquidity periods

name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels for 4h (using previous 4h bar's OHLC)
    # Camarilla R3 = close + 1.1*(high-low)/2
    # Camarilla S3 = close - 1.1*(high-low)/2
    # Using previous completed 4h bar to avoid look-ahead
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    camarilla_r3_4h = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 2
    camarilla_s3_4h = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 2
    
    # Align Camarilla levels to 1h
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA50 and volume MA)
    start_idx = 60  # max(50 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 4h EMA50
        uptrend = close[i] > ema50_4h_aligned[i]
        downtrend = close[i] < ema50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 in uptrend with volume confirmation
            if uptrend and close[i] > r3_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 in downtrend with volume confirmation
            elif downtrend and close[i] < s3_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price breaks below S3 (mean reversion) or trend changes
            if close[i] < s3_4h_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price breaks above R3 (mean reversion) or trend changes
            if close[i] > r3_4h_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals