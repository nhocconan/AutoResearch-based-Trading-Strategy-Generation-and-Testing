#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3, 1d EMA34 uptrend, and volume > 2.0x 20-bar avg.
# Short when price breaks below S3, 1d EMA34 downtrend, and volume > 2.0x 20-bar avg.
# Exit on touch of S3 (for longs) or R3 (for shorts) to capture mean reversion in ranging markets.
# Camarilla pivots provide precise intraday support/resistance levels that work well on 4h timeframe.
# Combined with 1d EMA34 trend filter to avoid counter-trend trades and volume confirmation to reduce false breakouts.
# Timeframe: 4h as per experiment guidelines.

name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (use previous day's high/low/close)
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    # We use the prior completed 1-day period to avoid look-ahead
    high_shift = df_1d['high'].shift(1).values
    low_shift = df_1d['low'].shift(1).values
    close_shift = df_1d['close'].shift(1).values
    
    # Align the prior day's OHLC to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_shift)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_shift)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_shift)
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_range = high_1d_aligned - low_1d_aligned
    r3 = close_1d_aligned + camarilla_range * 1.1 / 4
    s3 = close_1d_aligned - camarilla_range * 1.1 / 4
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = r3[i]
        curr_s3 = s3[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, uptrend (close > 1d EMA34), volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, downtrend (close < 1d EMA34), volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches or goes below S3 (mean reversion)
            if curr_close <= curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches or goes above R3 (mean reversion)
            if curr_close >= curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals