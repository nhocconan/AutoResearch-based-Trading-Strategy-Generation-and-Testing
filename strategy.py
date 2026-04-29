#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Uses 4h/1d for signal direction, 1h only for entry timing to avoid overtrading.
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume > 1.8x 24-bar avg
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume > 1.8x 24-bar avg
# Exit when price retouches Camarilla pivot point (mean reversion) or opposite breakout occurs
# Discrete position sizing (0.20) minimizes fee drag. Target: 15-37 trades/year on 1h.
# Session filter (08-20 UTC) reduces noise trades. 4h EMA50 ensures trend alignment.
# Volume confirmation ensures breakouts have conviction, reducing false signals.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Camarilla calculations
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_val = prior_high - prior_low
    camarilla_r3 = pivot + (range_val * 1.1)
    camarilla_s3 = pivot - (range_val * 1.1)
    camarilla_pivot = pivot  # Camarilla pivot point for exit
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: >1.8x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 24)  # Volume MA(24) and other indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pivot_pt = camarilla_pivot_aligned[i]
        ema_50 = ema_50_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume confirmation
            if curr_high > r3 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume confirmation
            elif curr_low < s3 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches Camarilla pivot or breaks below Camarilla S3
            if curr_close <= pivot_pt or curr_low < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price retouches Camarilla pivot or breaks above Camarilla R3
            if curr_close >= pivot_pt or curr_high > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals