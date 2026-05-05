#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when: price breaks above 4h Camarilla R3 AND 1d EMA34 shows uptrend (close > EMA34) AND volume > 1.5x 20-period MA
# Short when: price breaks below 4h Camarilla S3 AND 1d EMA34 shows downtrend (close < EMA34) AND volume > 1.5x 20-period MA
# Exit when: price returns to 4h Camarilla pivot point (PP) OR opposite breakout occurs
# Uses Camarilla for structure, 1d EMA for HTF trend, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels on 4h using daily lookback
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        # Use prior day's OHLC for Camarilla calculation
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        daily_range = prev_high - prev_low
        
        camarilla_pp = prev_close
        camarilla_r3 = prev_close + (daily_range * 1.1 / 4)
        camarilla_s3 = prev_close - (daily_range * 1.1 / 4)
    else:
        camarilla_pp = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    # Camarilla breakout signals
    camarilla_breakout_up = (close > camarilla_r3) & (np.roll(close, 1) <= np.roll(camarilla_r3, 1))
    camarilla_breakout_down = (close < camarilla_s3) & (np.roll(close, 1) >= np.roll(camarilla_s3, 1))
    camarilla_revert_pp = np.abs(close - camarilla_pp) < 0.001 * close  # approximate pivot point return
    
    # Get 1d data ONCE before loop for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough data for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Uptrend: close > EMA34, Downtrend: close < EMA34
    daily_uptrend = df_1d['close'].values > ema_34_1d
    daily_downtrend = df_1d['close'].values < ema_34_1d
    
    # Align 1d EMA trend to 4h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pp[i]) or 
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Camarilla breakout up + daily uptrend + volume filter
            if (camarilla_breakout_up[i] and 
                daily_uptrend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Camarilla breakout down + daily downtrend + volume filter
            elif (camarilla_breakout_down[i] and 
                  daily_downtrend_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla PP OR short breakout occurs
            if (camarilla_revert_pp[i] or camarilla_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla PP OR long breakout occurs
            if (camarilla_revert_pp[i] or camarilla_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals