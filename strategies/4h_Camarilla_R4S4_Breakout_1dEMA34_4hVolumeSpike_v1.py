#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and 4h volume spike confirmation.
# Long when price breaks above R4 with price > 1d EMA34 (bullish trend) and 4h volume > 1.8x 24-period average.
# Short when price breaks below S4 with price < 1d EMA34 (bearish trend) and 4h volume > 1.8x 24-period average.
# Exit on opposite Camarilla level (S4 for longs, R4 for shorts).
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume spike confirmation (1.8x) reduces false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe.
# Camarilla R4/S4 levels are stronger breakout levels than R3/S3, leading to fewer but higher quality trades.
# EMA34 on 1d provides smooth trend filter responsive enough for 4h breaks but resistant to whipsaw.

name = "4h_Camarilla_R4S4_Breakout_1dEMA34_4hVolumeSpike_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 1.8x 24-period average (tight filter to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike_4h = volume > (1.8 * vol_ma_24)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter (responsive yet smooth for 4h trading)
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # --- 4h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Vectorized prior day OHLC mapping for each 4h bar
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    # Precompute OHLC by date
    df_1d_pivot = df_1d_pivot.copy()
    df_1d_pivot['date'] = df_1d_pivot['open_time'].dt.date
    ohlc_map = df_1d_pivot.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    prior_day_dates = prior_day_start.dt.date
    high_vals = prior_day_dates.map(ohlc_map['high'])
    low_vals = prior_day_dates.map(ohlc_map['low'])
    close_vals = prior_day_dates.map(ohlc_map['close'])
    
    # Calculate Camarilla levels
    range_vals = high_vals - low_vals
    camarilla_r4 = close_vals + (range_vals * 1.1 / 2)  # R4
    camarilla_s4 = close_vals - (range_vals * 1.1 / 2)  # S4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or
            np.isnan(camarilla_r4[i]) or
            np.isnan(camarilla_s4[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 + price > 1d EMA34 (bullish) + 4h volume spike
            if (close[i] > camarilla_r4[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 + price < 1d EMA34 (bearish) + 4h volume spike
            elif (close[i] < camarilla_s4[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4
            if close[i] < camarilla_s4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4
            if close[i] > camarilla_r4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals