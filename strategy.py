#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour Camarilla R3/S3 breakout with 1-day EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses back inside Camarilla H3/L3 levels (mean reversion of breakout failure)
# Uses 4h/1d for signal direction (Camarilla pivots + EMA trend), 1h only for entry timing precision
# Session filter (08-20 UTC) to reduce noise trades
# Discrete position sizing (0.20) to minimize fee drag. Target: 15-37 trades/year on 1h.
# Camarilla pivots provide mathematically derived support/resistance levels that work in all market regimes.
# Volume confirmation ensures breakouts have conviction, reducing false signals in ranging markets.
# 1d EMA34 filter ensures we only trade with the higher timeframe trend, avoiding counter-trend whipsaws.

name = "1h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least one completed 4h bar
        return np.zeros(n)
    
    # Calculate Camarilla pivots on 4h data (using previous completed 4h bar)
    # Camarilla levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # R3 = close + 1.1*(high-low)/2 * 1.1/2, S3 = close - 1.1*(high-low)/2 * 1.1/2
    # Actually: R4 = close + 1.1*(high-low)/2 * 1.1, R3 = close + 1.1*(high-low)/2 * 0.55
    # Simpler: range = high - low, R3 = close + range * 1.1 * 0.55, S3 = close - range * 1.1 * 0.55
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    range_4h = high_4h - low_4h
    camarilla_r3_4h = close_4h + range_4h * 1.1 * 0.55
    camarilla_s3_4h = close_4h - range_4h * 1.1 * 0.55
    
    # Also calculate H3/L3 for exit levels (closer to price)
    camarilla_h3_4h = close_4h + range_4h * 1.1 * 0.275
    camarilla_l3_4h = close_4h - range_4h * 1.1 * 0.275
    
    # Align Camarilla levels to 1h timeframe (using completed 4h bars only)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 1h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need sufficient history for volume MA and EMA alignment
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]) or
            np.isnan(camarilla_h3_4h_aligned[i]) or np.isnan(camarilla_l3_4h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_4h_aligned[i]
        s3_level = camarilla_s3_4h_aligned[i]
        h3_level = camarilla_h3_4h_aligned[i]
        l3_level = camarilla_l3_4h_aligned[i]
        curr_close = close[i]
        prev_close = close[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume confirmation
            if curr_close > r3_level and prev_close <= r3_level and curr_close > ema_trend and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume confirmation
            elif curr_close < s3_level and prev_close >= s3_level and curr_close < ema_trend and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses back inside Camarilla H3/L3
            if curr_close < h3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price crosses back inside Camarilla H3/L3
            if curr_close > l3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals