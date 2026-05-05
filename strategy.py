#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d volume spike and 1d EMA50 trend filter
# Long when: price breaks above 4h R3, 1h volume > 2x 20-period average, and close > 1d EMA50
# Short when: price breaks below 4h S3, 1h volume > 2x 20-period average, and close < 1d EMA50
# Exit when price returns to 4h Camarilla R3/S3 level (mean reversion) or opposite breakout
# Uses 4h Camarilla levels for structure and 1d for trend/volume filters, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 1h, HTF: 4h/1d. Target: 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC) to reduce noise.

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_Session"
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
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 trend filter (for additional confirmation)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume average for spike detection
    df_1d_vol = get_htf_data(prices, '1d')  # Reload for volume (could optimize but clear)
    if len(df_1d_vol) >= 20:
        vol_1d = df_1d_vol['volume'].values
        vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = align_htf_to_ltf(prices, df_1d_vol, vol_1d > (2.0 * vol_ma_20_1d))
    else:
        volume_spike_1d = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels from previous 4h bar (using 1d OHLC for structure)
    if len(high_4h) >= 2:
        prev_high = np.roll(high_4h, 1)
        prev_low = np.roll(low_4h, 1)
        prev_close = np.roll(close_4h, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * rang * 1.1 / 4
        camarilla_s3 = prev_close - 1.1 * rang * 1.1 / 4
    else:
        camarilla_r3 = np.full(len(close_4h), np.nan)
        camarilla_s3 = np.full(len(close_4h), np.nan)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_filter[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 4h R3, volume filter, above 1d EMA50, and 1d volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                open_price[i] <= camarilla_r3_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume_spike_1d[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below 4h S3, volume filter, below 1d EMA50, and 1d volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  open_price[i] >= camarilla_s3_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume_spike_1d[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below 4h R3 (mean reversion) or breaks below 4h S3 (reversal)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above 4h S3 (mean reversion) or breaks above 4h R3 (reversal)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals