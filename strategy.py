#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA trend filter and 4h volume spike filter.
# Long when price breaks above R3 with 12h EMA(50) bullish (close > EMA) and 4h volume > 2.0x 20-period average.
# Short when price breaks below S3 with 12h EMA(50) bearish (close < EMA) and 4h volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike filter to reduce false breakouts.
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe.
# Works in bull/bear: 12h EMA ensures trend alignment, Camarilla R3/S3 provides tight structure, volume spike confirms momentum.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_4hVolumeSpike"
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
    # 4h volume spike: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.0 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_bullish = close_12h[-1] > ema_50_12h[-1] if len(close_12h) > 0 else False  # placeholder for alignment check
    # Actually use aligned EMA for trend: bullish when price > EMA
    # We'll compute trend inside loop using aligned arrays
    
    # --- 4h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 4h bar
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    for i in range(n):
        pd_ts = prior_day_start.iloc[i]
        day_mask = (df_1d_pivot['open_time'] >= pd_ts) & (df_1d_pivot['open_time'] < pd_ts + pd.Timedelta(days=1))
        if day_mask.any():
            day_data = df_1d_pivot.loc[day_mask]
            high_val = day_data['high'].iloc[0]
            low_val = day_data['low'].iloc[0]
            close_val = day_data['close'].iloc[0]
            range_val = high_val - low_val
            camarilla_r3[i] = close_val + (range_val * 1.1 / 4)  # R3
            camarilla_s3[i] = close_val - (range_val * 1.1 / 4)  # S3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + 12h EMA bullish (close > EMA) + 4h volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 12h EMA bearish (close < EMA) + 4h volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike_4h[i]):
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