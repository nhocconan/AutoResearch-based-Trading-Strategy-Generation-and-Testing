#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d EMA50 trend filter
# Long when price breaks above R3 AND volume > 2.0x 20-period average AND 1d EMA50 > EMA50_prev (uptrend)
# Short when price breaks below S3 AND volume > 2.0x 20-period average AND 1d EMA50 < EMA50_prev (downtrend)
# Exit when price crosses back to H3/L3 level OR 1d EMA50 flips direction
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-37 trades/year per symbol.
# Camarilla levels provide intraday support/resistance, volume spike confirms institutional interest,
# 1d EMA50 filters for primary trend direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Session filter (08-20 UTC) reduces noise trades during low-volume periods.

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-market to post-market overlap)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data ONCE before loop for Camarilla levels calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d data (using previous day's OHLC)
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + (rang * 1.1 / 4)
    camarilla_l3 = prev_close - (rang * 1.1 / 4)
    camarilla_h4 = prev_close + (rang * 1.1 / 2)
    camarilla_l4 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 4h data for volume spike confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Volume spike: 4h volume > 2.0x 20-period average
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = vol_4h > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h.astype(float))
    
    # Get 1d data for EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA50 > previous EMA50
    uptrend_1d = ema_50 > ema_50_prev
    downtrend_1d = ema_50 < ema_50_prev
    
    # Align 1d trend to 1h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 AND volume spike AND 1d uptrend
            if (close[i] > h3_aligned[i] and 
                volume_spike_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below L3 AND volume spike AND 1d downtrend
            elif (close[i] < l3_aligned[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back to L3 OR 1d trend flips to downtrend
            if (close[i] < l3_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back to H3 OR 1d trend flips to uptrend
            if (close[i] > h3_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals