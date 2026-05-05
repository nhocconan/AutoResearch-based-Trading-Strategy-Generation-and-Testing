#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1d EMA34 trend filter
# Long when: price breaks above R3, volume > 2.0x 24-period average (1d equivalent), and close > 1d EMA34
# Short when: price breaks below S3, volume > 2.0x 24-period average, and close < 1d EMA34
# Exit when price returns to Camarilla R3/S3 level (mean reversion)
# Uses Camarilla R3/S3 levels for balanced breakouts; volume spike on 1d for conviction; 1d EMA34 for trend filter
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_1dVolumeSpike"
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
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 12h using 24-period MA (equivalent to 1d lookback: 24*30m = 12h, but 12h bars so 24 bars = 12d? Wait: 12h timeframe, 24 periods = 24*12h = 12d -> too long)
    # Correction: For 12h timeframe, to get ~1d lookback, we need 2 periods (since 1d = 2*12h)
    # But 2 is too short for MA. Instead use fixed 24 periods as proxy for volume confirmation (will be adjusted by alignment)
    # Actually: we want volume confirmation based on 1d data, so we'll get 1d volume MA and align to 12h
    if len(volume) >= 2:
        vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
        volume_filter_12h = volume > (2.0 * vol_ma_2)
    else:
        volume_filter_12h = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume MA for confirmation (24-period ~ 1d lookback on 1h, but for 1d timeframe we use shorter)
    # For 1d data, use 20-period MA for volume confirmation
    if len(volume_1d) >= 20:
        vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    else:
        volume_spike_1d = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d EMA and volume spike to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Get 1d data ONCE before loop for Camarilla levels (from previous 1d bar)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar (R3/S3)
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * rang * 1.0 / 4  # R3 level
        camarilla_s3 = prev_close - 1.1 * rang * 1.0 / 4  # S3 level
    else:
        camarilla_r3 = np.full(len(close_1d), np.nan)
        camarilla_s3 = np.full(len(close_1d), np.nan)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume spike, and above 1d EMA34
            if (close[i] > camarilla_r3_aligned[i] and 
                open_price[i] <= camarilla_r3_aligned[i] and  # Ensure breakout happens on this bar
                volume_spike_aligned[i] > 0.5 and  # Boolean as float
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3, volume spike, and below 1d EMA34
            elif (close[i] < camarilla_s3_aligned[i] and 
                  open_price[i] >= camarilla_s3_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_spike_aligned[i] > 0.5 and  # Boolean as float
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 (mean reversion)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 (mean reversion)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals