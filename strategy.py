#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (EMA50) and 1d volume spike confirmation
# Long when: price breaks above 1h Camarilla R3 AND 4h EMA50 shows uptrend (close > EMA50) AND 1d volume > 2x 20-period MA
# Short when: price breaks below 1h Camarilla S3 AND 4h EMA50 shows downtrend (close < EMA50) AND 1d volume > 2x 20-period MA
# Exit when: price returns to 1h Camarilla pivot point (PP) OR opposite breakout occurs
# Uses Camarilla for precise intraday levels, 4h EMA for trend filter, 1d volume for conviction
# Timeframe: 1h, HTF: 4h/1d. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Session filter: 08-20 UTC to reduce noise trades outside active market hours.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Camarilla pivot levels from prior 1h OHLC (using current bar's data for next bar's levels)
    # Camarilla: PP = (H+L+C)/3, R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    typical_price = (high + low + close) / 3.0
    daily_range = high - low
    camarilla_pp = typical_price  # simplified: using typical price as pivot
    camarilla_r3 = close + (daily_range * 1.1 / 4)
    camarilla_s3 = close - (daily_range * 1.1 / 4)
    
    # Donchian-style breakout: price breaks above/below previous bar's R3/S3
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_pp_prev = np.roll(camarilla_pp, 1)
    
    camarilla_breakout_up = (close > camarilla_r3_prev) & (np.roll(close, 1) <= np.roll(camarilla_r3_prev, 1))
    camarilla_breakout_down = (close < camarilla_s3_prev) & (np.roll(close, 1) >= np.roll(camarilla_s3_prev, 1))
    camarilla_revert_pp = np.abs(close - camarilla_pp_prev) < 0.001 * close  # approximate pivot point return
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # need enough data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    close_4h = df_4h['close'].values
    if len(close_4h) >= 50:
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        # Bullish trend: 4h close > EMA50, Bearish trend: 4h close < EMA50
        trend_bullish = close_4h > ema_50_4h
        trend_bearish = close_4h < ema_50_4h
    else:
        ema_50_4h = np.full(len(close_4h), np.nan)
        trend_bullish = np.full(len(close_4h), False)
        trend_bearish = np.full(len(close_4h), False)
    
    # Align 4h EMA50 trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish.astype(float))
    
    # Get 1d data ONCE before loop for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # need enough data for volume MA
        return np.zeros(n)
    
    # Calculate volume confirmation on 1d using 20-period MA
    volume_1d = df_1d['volume'].values
    if len(volume_1d) >= 20:
        vol_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
        volume_spike = volume_1d > (2.0 * vol_ma_20_1d)
    else:
        vol_ma_20_1d = np.full(len(volume_1d), np.nan)
        volume_spike = np.zeros(len(volume_1d), dtype=bool)
    
    # Align 1d volume spike to 1h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_prev[i]) or np.isnan(camarilla_s3_prev[i]) or 
            np.isnan(camarilla_pp_prev[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Camarilla breakout up + 4h uptrend + 1d volume spike
            if (camarilla_breakout_up[i] and 
                trend_bullish_aligned[i] == 1.0 and 
                volume_spike_aligned[i] == 1.0):
                signals[i] = 0.20
                position = 1
            # Short conditions: Camarilla breakout down + 4h downtrend + 1d volume spike
            elif (camarilla_breakout_down[i] and 
                  trend_bearish_aligned[i] == 1.0 and 
                  volume_spike_aligned[i] == 1.0):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla PP OR short breakout occurs
            if (camarilla_revert_pp[i] or camarilla_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla PP OR long breakout occurs
            if (camarilla_revert_pp[i] or camarilla_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals