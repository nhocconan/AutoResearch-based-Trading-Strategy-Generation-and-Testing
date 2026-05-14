#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w HMA21 trend filter and 1d volume spike confirmation.
# Long when price breaks above R3 with price > 1w HMA21 (bullish trend) and 1d volume > 2.0x 20-period average.
# Short when price breaks below S3 with price < 1w HMA21 (bearish trend) and 1d volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses 1d timeframe for lower trade frequency (target: 30-100 total trades over 4 years) and 1w HMA21 for strong trend filter.
# Volume spike confirmation (2.0x) reduces false breakouts. Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets.

name = "1d_Camarilla_R3S3_Breakout_1wHMA21_1dVolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # 1d volume confirmation: > 2.0x 20-period average (stricter filter to reduce trade frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume > (2.0 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w HMA(21) - trend filter (Hull Moving Average for reduced lag)
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    wma_half = wma(close_1w, half_n)
    wma_full = wma(close_1w, 21)
    wma_diff = 2 * wma_half - wma_full
    hma_21 = wma(wma_diff, sqrt_n)
    
    # Pad HMA to match original length
    hma_21_padded = np.full(len(close_1w), np.nan)
    hma_21_padded[half_n + sqrt_n - 1:] = hma_21
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
    
    # --- 1d Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 1d bar using vectorized approach
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    # Merge to get prior day's OHLC for each timestamp
    df_1d_pivot = df_1d_pivot.copy()
    df_1d_pivot['date'] = df_1d_pivot['open_time'].dt.date
    prior_day_start_date = prior_day_start.dt.date
    
    # Create mapping from date to OHLC
    ohlc_map = df_1d_pivot.groupby('date').agg({
        'high': 'first',
        'low': 'first',
        'close': 'first'
    })
    
    for i in range(n):
        pd_date = prior_day_start_date.iloc[i]
        if pd_date in ohlc_map.index:
            day_data = ohlc_map.loc[pd_date]
            high_val = day_data['high']
            low_val = day_data['low']
            close_val = day_data['close']
            range_val = high_val - low_val
            camarilla_r3[i] = close_val + (range_val * 1.1 / 2)  # R3
            camarilla_s3[i] = close_val - (range_val * 1.1 / 2)  # S3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(hma_21_aligned[i]) or
            np.isnan(volume_spike_1d[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + price > 1w HMA21 (bullish) + 1d volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_spike_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price < 1w HMA21 (bearish) + 1d volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < hma_21_aligned[i] and 
                  volume_spike_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals