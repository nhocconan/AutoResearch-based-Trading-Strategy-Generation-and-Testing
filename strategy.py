#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide strong intraday support/resistance that work in both bull and bear markets.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts. Designed for 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via upward breaks at R3/R4 and in bear markets via downward breaks at S3/S4.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get previous day's OHLC for Camarilla calculation
    # We need to resample to get daily OHLC from the 4h data (but using actual Binance boundaries)
    # Since we're on 4h timeframe, we can get daily data directly
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    # Camarilla R4 = close + (high - low) * 1.1/2
    # Camarilla S4 = close - (high - low) * 1.1/2
    
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    
    # Align the Camarilla levels to 4h timeframe
    camarilla_r3 = align_htf_to_ltf(prices, df_1d_ohlc, prev_close + (prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = align_htf_to_ltf(prices, df_1d_ohlc, prev_close - (prev_high - prev_low) * 1.1 / 4)
    camarilla_r4 = align_htf_to_ltf(prices, df_1d_ohlc, prev_close + (prev_high - prev_low) * 1.1 / 2)
    camarilla_s4 = align_htf_to_ltf(prices, df_1d_ohlc, prev_close - (prev_high - prev_low) * 1.1 / 2)
    
    # Volume confirmation: 20-period EMA on 4h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above camarilla R3 in uptrend alignment with volume spike
            if close[i] > camarilla_r3[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below camarilla S3 in downtrend alignment with volume spike
            elif close[i] < camarilla_s3[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below camarilla S3 or loses uptrend alignment
            if close[i] < camarilla_s3[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above camarilla R3 or loses downtrend alignment
            if close[i] > camarilla_r3[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals