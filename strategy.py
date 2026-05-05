#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND close > EMA50(4h) AND volume > 2.0x 20-period average
# Short when price breaks below S3 AND close < EMA50(4h) AND volume > 2.0x 20-period average
# Exit when price re-enters the Camarilla H-L range OR EMA50(4h) trend flips
# Uses 1h primary timeframe with 4h HTF for trend filter to reduce whipsaw
# Uses 1d HTF for Camarilla pivot calculation (standard for intraday)
# Discrete sizing (0.20) to limit fee drag and manage drawdown
# Session filter: 08-20 UTC to reduce noise trades
# Works in both bull and bear markets by adapting to trend via EMA filter
# Target: 15-37 trades/year (60-150 over 4 years) to avoid fee drag

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_Session"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot calculation (standard for intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: H-L range, then R3/S3 = close ± (H-L)*1.1/4
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (will be available after 1d bar closes)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND close > EMA50(4h) AND volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND close < EMA50(4h) AND volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla H-L range OR close < EMA50(4h) (trend flip)
            if (low[i] < camarilla_r3_aligned[i] and high[i] > camarilla_s3_aligned[i]) or \
               close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters Camarilla H-L range OR close > EMA50(4h) (trend flip)
            if (low[i] < camarilla_r3_aligned[i] and high[i] > camarilla_s3_aligned[i]) or \
               close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals