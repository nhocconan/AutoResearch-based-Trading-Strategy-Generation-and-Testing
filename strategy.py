#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when: price breaks above 12h Camarilla R3 AND 1d EMA34 is rising (trend up) AND volume > 2x 20-period MA
# Short when: price breaks below 12h Camarilla S3 AND 1d EMA34 is falling (trend down) AND volume > 2x 20-period MA
# Exit when: price returns to 12h Camarilla pivot point (PP) OR opposite breakout occurs
# Uses Camarilla for structure, 1d EMA for trend bias, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels on 12h (using daily formula but applied to 12h bars)
    # Camarilla: PP = (high + low + close)/3
    #          R3 = close + ((high-low) * 1.1/4)
    #          S3 = close - ((high-low) * 1.1/4)
    if len(high) >= 1 and len(low) >= 1 and len(close) >= 1:
        camarilla_pp = (high + low + close) / 3.0
        daily_range = high - low
        camarilla_r3 = close + (daily_range * 1.1 / 4)
        camarilla_s3 = close - (daily_range * 1.1 / 4)
    else:
        camarilla_pp = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    # Camarilla breakout signals
    camarilla_breakout_up = (close > camarilla_r3) & (np.roll(close, 1) <= np.roll(camarilla_r3, 1))
    camarilla_breakout_down = (close < camarilla_s3) & (np.roll(close, 1) >= np.roll(camarilla_s3, 1))
    camarilla_revert_pp = np.abs(close - camarilla_pp) < 0.001 * close  # approximate PP return
    
    # Get 1d data ONCE before loop for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.diff(ema_34_1d, prepend=ema_34_1d[0]) > 0  # True when rising
    ema_34_falling = np.diff(ema_34_1d, prepend=ema_34_1d[0]) < 0  # True when falling
    
    # Align 1d EMA34 trend to 12h timeframe
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising.astype(float))
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Camarilla breakout up + 1d EMA34 rising + volume filter
            if (camarilla_breakout_up[i] and 
                ema_34_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Camarilla breakout down + 1d EMA34 falling + volume filter
            elif (camarilla_breakout_down[i] and 
                  ema_34_falling_aligned[i] == 1.0 and 
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