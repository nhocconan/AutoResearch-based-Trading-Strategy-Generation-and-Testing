#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 4h volume spike confirmation.
# Long when price breaks above R1 with price > 4h EMA50 (bullish trend) and 4h volume > 2.0x 20-period average.
# Short when price breaks below S1 with price < 4h EMA50 (bearish trend) and 4h volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses 4h HTF for trend to reduce noise and overtrading vs 1d. Volume spike confirmation (2.0x) reduces false breakouts.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h. HARD MAX: 200 total trades.
# Session filter: 08-20 UTC to avoid low-liquidity hours.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_4hVolumeSpike_v1"
timeframe = "1h"
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
    
    # --- 1h Indicators (LTF) ---
    # 1h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume > (2.0 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) - trend filter
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # --- 1h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 1h bar using vectorized approach
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
            camarilla_r1[i] = close_val + (range_val * 1.1 / 4)  # R1
            camarilla_s1[i] = close_val - (range_val * 1.1 / 4)  # S1
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_1h[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            not (8 <= hours[i] <= 20)):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + price > 4h EMA50 (bullish) + 1h volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_1h[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + price < 4h EMA50 (bearish) + 1h volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_1h[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals