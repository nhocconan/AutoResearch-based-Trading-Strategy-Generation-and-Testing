#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when: price breaks above 12h Camarilla R3 AND 1d EMA34 shows uptrend (close > EMA34) AND volume > 2x 20-period MA
# Short when: price breaks below 12h Camarilla S3 AND 1d EMA34 shows downtrend (close < EMA34) AND volume > 2x 20-period MA
# Exit when: price returns to 12h Camarilla pivot point (PP) OR opposite breakout occurs
# Uses Camarilla for precise pivot levels, 1d EMA for trend filter, volume for conviction
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
    
    # Calculate Camarilla levels on 12h using prior bar's OHLC
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        # Use prior bar's OHLC for today's Camarilla levels (no look-ahead)
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        daily_range = prev_high - prev_low
        camarilla_pp = prev_close  # Pivot Point = previous close
        camarilla_r3 = camarilla_pp + (daily_range * 1.1 / 4)
        camarilla_s3 = camarilla_pp - (daily_range * 1.1 / 4)
    else:
        camarilla_pp = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    # Camarilla breakout signals
    camarilla_breakout_up = (close > camarilla_r3) & (np.roll(close, 1) <= np.roll(camarilla_r3, 1))
    camarilla_breakout_down = (close < camarilla_s3) & (np.roll(close, 1) >= np.roll(camarilla_s3, 1))
    camarilla_revert_pp = np.abs(close - camarilla_pp) < 0.001 * close  # approximate pivot point return
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Bullish trend: close > EMA34, Bearish trend: close < EMA34
    daily_bullish = df_1d['close'].values > ema_34_1d
    daily_bearish = df_1d['close'].values < ema_34_1d
    
    # Align 1d EMA34 trend to 12h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_pp[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Camarilla breakout up + daily bullish + volume filter
            if (camarilla_breakout_up[i] and 
                daily_bullish_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Camarilla breakout down + daily bearish + volume filter
            elif (camarilla_breakout_down[i] and 
                  daily_bearish_aligned[i] == 1.0 and 
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